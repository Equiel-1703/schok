{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
module Main where

import qualified Data.Array.Accelerate          as A
import           Data.Array.Accelerate            ( Acc, Vector, Scalar, Z(..) )
import qualified Data.Array.Accelerate.LLVM.PTX as PTX
import           Control.DeepSeq    ( force )
import           Control.Exception  ( evaluate )
import           Data.Time.Clock    ( getCurrentTime, diffUTCTime )
import           System.Environment ( getArgs )
import           System.Random      ( mkStdGen, randomRs )
import           Text.Printf        ( printf )

-- Same value ranges as the CUDA loadData (not bit-identical to C's rand()).
genLocations :: Int -> [(Float, Float)]
genLocations n =
  let lats = take n (randomRs (7.0, 70.0)  (mkStdGen 42))
      lngs = take n (randomRs (0.0, 358.0) (mkStdGen 1337))
  in  zip lats lngs

-- Builds and fully forces the host-side input (done outside the timed region).
mkInput :: Int -> IO (Vector (Float, Float))
mkInput n = do
  let xs = genLocations n
  _ <- evaluate (force xs)
  let arr = A.fromList (Z A.:. n) xs :: Vector (Float, Float)
  _ <- evaluate arr
  return arr

-- map: Euclidean distance to the origin; fold: minimum.
nearestNeighbor :: Acc (Vector (Float, Float)) -> Acc (Scalar Float)
nearestNeighbor locs =
  A.fold1All A.min (A.map dist locs)
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
         []    -> error "usage: nn-warm <numRecords>"

  input <- mkInput n

  -- Warm-up (NOT timed): same program on a tiny input, so the GPU context,
  -- memory manager and kernel compilation are done before measuring.
  let f = PTX.runN nearestNeighbor :: Vector (Float, Float) -> Scalar Float
  warmInput <- mkInput 1024
  _ <- evaluate (A.indexArray (f warmInput) Z)

  t0 <- getCurrentTime
  let result = f input
  minDist <- evaluate (A.indexArray result Z)
  t1 <- getCurrentTime

  let ms = realToFrac (diffUTCTime t1 t0) * 1000 :: Double
  printf "Accelerate-warm\t%d\t%.1f\n" n ms
  printf "min distance = %f\n" minDist
