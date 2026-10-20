{-# LANGUAGE FlexibleContexts #-}

-- Double-precision version of Main.hs: single execution, NO warm-up.
-- `run` is called exactly once, so the timed region includes Accelerate's
-- one-off costs (backend/context init, JIT or cache load of the kernel).

module Main where

import Data.Array.Accelerate                 as A
import Data.Array.Accelerate.LLVM.PTX         as PTX

import Control.DeepSeq                        (force)
import Control.Exception                      (evaluate)
import Data.Time.Clock                         (diffUTCTime, getCurrentTime)
import System.Environment                      (getArgs)
import System.Mem                              (performBlockingMajorGC)
import System.Random                           (newStdGen, randomRs)
import Text.Printf                             (printf)
import System.Mem (performBlockingMajorGC)

dotProduct :: Acc (Vector Double) -> Acc (Vector Double) -> Acc (Scalar Double)
dotProduct a b = A.fold (+) 0 (A.zipWith (*) a b)

-- N pseudo-random doubles with the magnitude of C's rand() (POSIX RAND_MAX),
-- matching a[i] = rand() with an implicit int->double conversion.
genDoubles :: Int -> IO [Double]
genDoubles n = do
  gen <- newStdGen
  return $ Prelude.map Prelude.fromIntegral
          $ Prelude.take n (randomRs (0 :: Int, 2147483647) gen)

main :: IO ()
main = do
  args <- getArgs
  n <- case args of
         (x:_) -> return (read x :: Int)
         _     -> error "usage: dotprod-double N"

  asL <- genDoubles n
  bsL <- genDoubles n
  _   <- evaluate (force asL)
  _   <- evaluate (force bsL)

  let vecA = A.fromList (Z :. n) asL :: Vector Double
      vecB = A.fromList (Z :. n) bsL :: Vector Double
  -- Force the list -> Array conversion now, so it is not counted below.
  _ <- evaluate (force vecA)
  _ <- evaluate (force vecB)

  -- Keep host-side garbage collection out of the timed region (the CUDA
  -- program has no GC).
  performBlockingMajorGC

  t0 <- getCurrentTime
  let result = PTX.run (dotProduct (A.use vecA) (A.use vecB))
  _ <- evaluate (force result)
  t1 <- getCurrentTime

  let elapsedMs = realToFrac (diffUTCTime t1 t0) * 1000 :: Double
  printf "Accelerate-double\t%d\t%3.1f\n" n elapsedMs
