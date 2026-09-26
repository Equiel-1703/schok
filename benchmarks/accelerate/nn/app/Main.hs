{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
module Main where

import qualified Data.Array.Accelerate          as A
import           Data.Array.Accelerate            ( Acc, Vector, Scalar, Z(..) )
import qualified Data.Array.Accelerate.LLVM.PTX as PTX

import           System.Environment               ( getArgs )
import           System.Random                     ( mkStdGen, randomRs )
import           Data.Time.Clock                   ( getCurrentTime, diffUTCTime )
import           Control.Exception                 ( evaluate )
import           Control.DeepSeq                   ( force )
import           Text.Printf                        ( printf )

-- ---------------------------------------------------------------------------
-- Data generation
--
-- This mirrors the *shape* and value range of the CUDA program's loadData:
--   lat = 7 + rand()%63 + fraction   -> roughly in [7, 70)
--   lng =     rand()%358 + fraction  -> roughly in [0, 358)
--
-- NOTE: this does NOT reproduce the exact bit-for-bit sequence that C's
-- rand() would produce (Haskell's `random` package uses a different PRNG),
-- so the two programs will not compute the minimum distance from identical
-- input data. That does not affect the fairness of a *runtime* comparison
-- here: both kernels do the same amount of work (one map + one min-reduce
-- over `n` records, transferred to/from the GPU) regardless of the actual
-- float values, so wall-clock time is not sensitive to which values were
-- generated. If you need bit-identical inputs (e.g. to cross-check the
-- numeric result, not just timing), generate a single locations file once
-- and modify both programs to read from it instead of generating data
-- themselves.
-- ---------------------------------------------------------------------------
genLocations :: Int -> [(Float, Float)]
genLocations n =
  let lats = take n (randomRs (7.0, 70.0)  (mkStdGen 42))
      lngs = take n (randomRs (0.0, 358.0) (mkStdGen 1337))
  in  zip lats lngs

-- | map: Euclidean distance from the origin (0,0) to every point
--   fold: minimum of all distances
-- This is the Accelerate equivalent of map_step_2para_1resp_kernel (with
-- par1 = par2 = 0.0, i.e. euclid against the origin) followed by
-- reduce_kernel using the `menor` (min) combinator.
nearestNeighbor :: Vector (Float, Float) -> Acc (Scalar Float)
nearestNeighbor locs =
  let dists = A.map dist (A.use locs)
  in  A.fold1All A.min dists
  where
    dist :: A.Exp (Float, Float) -> A.Exp Float
    dist p =
      let (x, y) = A.unlift p :: (A.Exp Float, A.Exp Float)
      in  A.sqrt (x * x + y * y)

main :: IO ()
main = do
  args <- getArgs
  n <- case args of
         (x:_) -> pure (read x)
         []    -> error "usage: nn-accelerate <numRecords>"

  -- Build the host-side input array BEFORE starting the timer, mirroring
  -- the CUDA program: loadData() runs, and only *then* is cudaEventRecord
  -- for `start` called. Host-side data preparation is deliberately
  -- excluded from both timings.
  let locsList = genLocations n
  _ <- evaluate (force locsList)
  let locsArr = A.fromList (Z A.:. n) locsList :: Vector (Float, Float)
  _ <- evaluate locsArr

  t0 <- getCurrentTime

  -- A single, cold call to `run`: this both moves data to the GPU and
  -- back, and executes the map+fold there. Because accelerate-llvm-ptx
  -- compiles the Accelerate program to PTX at run time, this one-shot
  -- timing also includes that JIT/codegen cost (there is no separate
  -- "warm up" call anywhere in this program) -- unlike the CUDA program,
  -- whose kernels were already compiled ahead of time by nvcc. That
  -- compilation-model difference is an inherent property of the two
  -- frameworks, not something this benchmark tries to hide.
  let result = PTX.run (nearestNeighbor locsArr)

  -- Force the scalar result so the device->host copy is actually
  -- included in the measured time, not deferred lazily past `t1`.
  minDist <- evaluate (A.indexArray result Z)

  t1 <- getCurrentTime
  let elapsedMs = realToFrac (diffUTCTime t1 t0) * 1000 :: Double

  printf "Accelerate\t%d\t%.1f\n" n elapsedMs
  printf "min distance = %f\n" minDist
