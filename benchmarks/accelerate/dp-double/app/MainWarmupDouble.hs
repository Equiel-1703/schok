{-# LANGUAGE FlexibleContexts #-}

-- Double-precision version of MainWarmup.hs.
-- The program is compiled once with `PTX.runN`, applied first to a small
-- dummy input (untimed warm-up: backend/context init, kernel compile or
-- cache load, module link), then applied to the real data (timed).
-- The warm-up uses a small fixed size rather than N so that Accelerate's
-- device memory cache is not pre-populated; the CUDA program pays for its
-- device allocations inside its timed window.

module Main where

import Data.Array.Accelerate                 as A
import Data.Array.Accelerate.LLVM.PTX         as PTX

import Control.DeepSeq                        (force)
import Control.Exception                      (evaluate)
import Data.Time.Clock                         (diffUTCTime, getCurrentTime)
import System.Environment                      (getArgs)
import System.IO                               (stderr)
import System.Mem                              (performBlockingMajorGC)
import System.Random                           (newStdGen, randomRs)
import Text.Printf                             (printf, hPrintf)
import System.Mem (performBlockingMajorGC)

dotProduct :: Acc (Vector Double) -> Acc (Vector Double) -> Acc (Scalar Double)
dotProduct a b = A.fold (+) 0 (A.zipWith (*) a b)

genDoubles :: Int -> IO [Double]
genDoubles n = do
  gen <- newStdGen
  return $ Prelude.map Prelude.fromIntegral
          $ Prelude.take n (randomRs (0 :: Int, 2147483647) gen)

warmupSize :: Int
warmupSize = 1024

main :: IO ()
main = do
  args <- getArgs
  n <- case args of
         (x:_) -> return (read x :: Int)
         _     -> error "usage: dotprod-double-warmup N"

  asL <- genDoubles n
  bsL <- genDoubles n
  _   <- evaluate (force asL)
  _   <- evaluate (force bsL)

  let vecA = A.fromList (Z :. n) asL :: Vector Double
      vecB = A.fromList (Z :. n) bsL :: Vector Double
  -- Force the list -> Array conversion now, so it is not counted below.
  _ <- evaluate (force vecA)
  _ <- evaluate (force vecB)

  let dot = PTX.runN dotProduct :: Vector Double -> Vector Double -> Scalar Double

  -- Warm-up (NOT timed): small dummy input, same compiled program.
  let wA = A.fromList (Z :. warmupSize) (Prelude.replicate warmupSize 1) :: Vector Double
      wB = A.fromList (Z :. warmupSize) (Prelude.replicate warmupSize 1) :: Vector Double

  -- GC before the warm-up too, so a collection of the huge input lists
  -- does not land inside it.
  performBlockingMajorGC
  w0 <- getCurrentTime
  _ <- evaluate (force (dot wA wB))
  w1 <- getCurrentTime
  -- Diagnostic only (stderr, so stdout stays one clean result line).
  hPrintf stderr "warmup-ms\t%3.1f\n"
          (realToFrac (diffUTCTime w1 w0) * 1000 :: Double)

  performBlockingMajorGC

  -- Timed run: transfer + compute + transfer back, no compilation.
  t0 <- getCurrentTime
  let result = dot vecA vecB
  _ <- evaluate (force result)
  t1 <- getCurrentTime

  let elapsedMs = realToFrac (diffUTCTime t1 t0) * 1000 :: Double
  printf "Accelerate-warmup-double\t%d\t%3.1f\n" n elapsedMs
