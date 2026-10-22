-- |
-- COLD version: no warm-up. The single timed call to `PTX.run` includes CUDA
-- context creation and JIT compilation. The program itself (`pixelsN`) is
-- shared with the warm executable via module Julia.
module Main where

import qualified Data.Array.Accelerate           as A
import           Data.Array.Accelerate           (Z (..))
import qualified Data.Array.Accelerate.LLVM.PTX  as PTX

import           Control.DeepSeq                 (force)
import           Control.Exception               (evaluate)
import           Data.Time.Clock                 (diffUTCTime, getCurrentTime)
import           System.Environment              (getArgs)
import           System.Exit                     (exitFailure)
import           System.Mem                      (performBlockingMajorGC)
import           Text.Printf                     (printf)

import           Julia                           (pixelsN)

main :: IO ()
main = do
  args <- getArgs
  case args of
    [s] -> runCold (read s)
    _   -> putStrLn "usage: accelerate-julia-cold <dim>" >> exitFailure

runCold :: Int -> IO ()
runCold dim = do
  performBlockingMajorGC

  t0 <- getCurrentTime

  let result = PTX.run (pixelsN (A.use (A.fromList Z [dim])))
  _ <- evaluate (force result)

  t1 <- getCurrentTime
  let millis = realToFrac (diffUTCTime t1 t0) * 1000 :: Double
  printf "Accelerate\t%d\t%.1f\n" dim millis

  -- Sanity check (do this in a separate, untimed run):
  -- print (A.toList result)
