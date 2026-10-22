{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE PatternSynonyms     #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators       #-}

-- |
-- WARM version of the Accelerate Julia-set benchmark.
--
-- The program is compiled once with `PTX.runN`. `dim` is a *runtime* input
-- (a Scalar Int) instead of a constant baked into the program, so one
-- compiled program serves any size. We first call it on a tiny input
-- (this triggers CUDA context creation, JIT compilation and the first
-- kernel load), then run the real size and time only that call.
--
-- The timed call still includes: uploading the scalar `dim`, device
-- allocation, kernel execution, and the device-to-host copy of the result.
-- It does not include JIT compilation.
module Main where

import qualified Data.Array.Accelerate           as A
import           Data.Array.Accelerate           (Acc, Array, DIM3, Exp, Scalar,
                                                    Z (..), (:.) (..))
import qualified Data.Array.Accelerate.LLVM.PTX  as PTX

import           Control.DeepSeq                 (force)
import           Control.Exception               (evaluate)
import           Data.Int                        (Int32)
import           Data.Time.Clock                 (diffUTCTime, getCurrentTime)
import           System.Environment              (getArgs)
import           System.Exit                     (exitFailure)
import           System.Mem                      (performBlockingMajorGC)
import           Text.Printf                     (printf)

--------------------------------------------------------------------------------
-- Julia set escape test (identical to the cold version)
--------------------------------------------------------------------------------

julia :: Exp Int -> Exp Int -> Exp Int -> Exp Int32
julia x y dim =
  let scale = 0.1 :: Exp Float
      dimF  = A.fromIntegral dim :: Exp Float
      jx    = (scale * (dimF - A.fromIntegral x)) / dimF
      jy    = (scale * (dimF - A.fromIntegral y)) / dimF
      cr    = -0.8  :: Exp Float
      ci    = 0.156 :: Exp Float

      cond :: Exp (Float, Float, Bool, Int) -> Exp Bool
      cond st =
        let A.T4 _ _ _ i = st
        in i A.< 200

      step :: Exp (Float, Float, Bool, Int) -> Exp (Float, Float, Bool, Int)
      step st =
        let A.T4 ar ai diverged i = st
            nar          = (ar * ar - ai * ai) + cr
            nai          = (ai * ar + ar * ai) + ci
            mag2         = nar * nar + nai * nai
            justDiverged = mag2 A.> 1.0e3
            diverged'    = diverged A.|| justDiverged
            ar'          = A.cond diverged ar nar
            ai'          = A.cond diverged ai nai
        in A.T4 ar' ai' diverged' (i + 1)

      initial = A.T4 jx jy (A.constant False) (0 :: Exp Int)
      final   = A.while cond step initial
      A.T4 _ _ divergedFinal _ = final
  in A.cond divergedFinal 0 1

--------------------------------------------------------------------------------
-- Pixel buffer generation, with dim as a runtime argument
--------------------------------------------------------------------------------

pixelsN :: Acc (Scalar Int) -> Acc (Array DIM3 Int32)
pixelsN dimA =
  A.generate (A.I3 dim dim 4) go
  where
    dim = A.the dimA
    go ix =
      let Z :. y :. x :. c = A.unlift ix :: Z :. Exp Int :. Exp Int :. Exp Int
          jv = julia x y dim
      in A.cond (c A.== 0) (255 * jv)
       $ A.cond (c A.== 3) 255
       $ 0

-- | Compiled once, shared by the warm-up call and the timed call.
pixelsRun :: Scalar Int -> Array DIM3 Int32
pixelsRun = PTX.runN pixelsN
{-# NOINLINE pixelsRun #-}

--------------------------------------------------------------------------------
-- main
--------------------------------------------------------------------------------

-- | Size used for the warm-up call. Compilation does not depend on it.
warmDim :: Int
warmDim = 64

main :: IO ()
main = do
  args <- getArgs
  case args of
    [s] -> runWarm (read s)
    _   -> putStrLn "usage: accelerate-julia-warm <dim>" >> exitFailure

runWarm :: Int -> IO ()
runWarm dim = do
  -- Warm-up (not timed): context creation + JIT compilation + kernel load.
  _ <- evaluate (force (pixelsRun (A.fromList Z [warmDim])))

  performBlockingMajorGC

  t0 <- getCurrentTime

  let result = pixelsRun (A.fromList Z [dim])
  _ <- evaluate (force result)

  t1 <- getCurrentTime
  let millis = realToFrac (diffUTCTime t1 t0) * 1000 :: Double
  printf "AccelerateWarm\t%d\t%.1f\n" dim millis

  -- Sanity check (do this in a separate, untimed run):
  -- print (A.toList result)
