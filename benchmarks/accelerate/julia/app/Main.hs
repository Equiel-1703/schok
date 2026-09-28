{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE PatternSynonyms     #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators       #-}

-- |
-- Accelerate re-implementation of the CUDA julia.cu program.
--
-- Computes a DIM x DIM Julia-set image, one pixel represented as four
-- consecutive Int32s (r,g,b,a) in a flat array, exactly mirroring the
-- memory layout AND element width produced by the CUDA kernel:
--
--     ptr[(x + y*dim)*4 + 0] = 255 * juliaValue(x,y)
--     ptr[(x + y*dim)*4 + 1] = 0
--     ptr[(x + y*dim)*4 + 2] = 0
--     ptr[(x + y*dim)*4 + 3] = 255
--
-- NOTE: element type is Int32 (4 bytes), matching CUDA's `int` exactly.
-- Using Accelerate's default `Int` (8 bytes on 64-bit systems) would
-- silently double the device memory footprint versus the CUDA version --
-- for dim=20000 that's 12.8GB vs CUDA's 6.4GB, enough to exhaust GPU
-- memory on one side and not the other, invalidating the comparison
-- (and precisely what caused the "Remote memory exhausted" crash).
--
-- Timing wraps the single call to `PTX.run`, which is where Accelerate
-- allocates device memory, JIT-compiles / launches the kernel, and
-- copies the result back to the host -- the GPU-side work the CUDA
-- program times with cudaEvent start/stop.
module Main where

import qualified Data.Array.Accelerate           as A
import           Data.Array.Accelerate           (Acc, Array, DIM3, Exp,
                                                    Z (..), (:.) (..))
import qualified Data.Array.Accelerate.LLVM.PTX  as PTX

import           Data.Int                        (Int32)
import           Control.Exception               (evaluate)
import           Data.Time.Clock                 (diffUTCTime, getCurrentTime)
import           System.Environment              (getArgs)
import           System.Exit                     (exitFailure)
import           Text.Printf                     (printf)

--------------------------------------------------------------------------------
-- Julia set escape test
--------------------------------------------------------------------------------

-- | Mirrors the CUDA __device__ function `julia`: returns 1 if the point
-- stays bounded for 200 iterations, 0 if it escapes (|z|^2 > 1000) at any
-- point during those 200 iterations.
--
-- Indices/loop counters stay as (machine) Int -- only the final pixel
-- array element type is narrowed to Int32, to match CUDA's `int` width.
--
-- Accelerate has no early `break` inside a device-side loop, so instead of
-- stopping the loop we track a `diverged` flag: once set, ar/ai are frozen
-- (not further updated), which is observationally identical to the CUDA
-- version's early return, while staying well-defined for exactly 200 steps.
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
-- Pixel buffer generation
--------------------------------------------------------------------------------

-- | Build the DIM x DIM x 4 pixel array. Row-major flattening of a
-- DIM3 = Z :. Int :. Int :. Int Accelerate array puts the last index
-- fastest, so element (y,x,c) sits at ((y*dim)+x)*4+c -- identical to
-- the CUDA kernel's ptr[(x + y*dim)*4 + c] layout. Element type is
-- Int32, matching CUDA's `int` (see note above).
pixels :: Int -> Acc (Array DIM3 Int32)
pixels dim =
  A.generate (A.constant (Z :. dim :. dim :. 4)) go
  where
    dim' = A.constant dim
    go ix =
      let Z :. y :. x :. c = A.unlift ix :: Z :. Exp Int :. Exp Int :. Exp Int
          jv = julia x y dim'
      in A.cond (c A.== 0) (255 * jv)
       $ A.cond (c A.== 3) 255
       $ 0

--------------------------------------------------------------------------------
-- main
--------------------------------------------------------------------------------

main :: IO ()
main = do
  args <- getArgs
  case args of
    [s] -> runOnce (read s)
    _   -> putStrLn "usage: accelerate-julia <dim>" >> exitFailure

runOnce :: Int -> IO ()
runOnce dim = do
  t0     <- getCurrentTime
  result <- evaluate (PTX.run (pixels dim))
  t1     <- getCurrentTime

  let millis = realToFrac (diffUTCTime t1 t0) * 1000 :: Double
  printf "Accelerate\t%d\t%.1f\n" dim millis

  -- Uncomment to sanity-check the output (do this in a *separate* run from
  -- the one you're timing, since building/printing the list is extra work
  -- that has no CUDA-side equivalent in the timed region):
  -- print (A.toList result)

  -- silence unused-variable warning when the line above stays commented
  result `seq` return ()
