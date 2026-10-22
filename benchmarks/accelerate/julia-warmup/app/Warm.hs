-- |
-- WARM version: the same program (`pixelsN`, shared via module Julia) is
-- compiled once with `PTX.runN`. A warm-up call on a tiny input triggers CUDA
-- context creation, JIT compilation and kernel load; only the second call,
-- at the real size, is timed.
module Main where

import qualified Data.Array.Accelerate           as A
import           Data.Array.Accelerate           (Array, DIM3, Scalar, Z (..))
import qualified Data.Array.Accelerate.LLVM.PTX  as PTX

import           Control.DeepSeq                 (force)
import           Control.Exception               (evaluate)
import           Data.Int                        (Int32)
import           Data.Time.Clock                 (diffUTCTime, getCurrentTime)
import           System.Environment              (getArgs)
import           System.Exit                     (exitFailure)
import           System.Mem                      (performBlockingMajorGC)
import           Text.Printf                     (printf)

import           Julia                           (pixelsN)

-- | Compiled once, shared by the warm-up call and the timed call.
pixelsRun :: Scalar Int -> Array DIM3 Int32
pixelsRun = PTX.runN pixelsN
{-# NOINLINE pixelsRun #-}

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
  -- Warm-up (not timed)
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
