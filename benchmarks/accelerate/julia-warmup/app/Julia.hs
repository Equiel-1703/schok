{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE PatternSynonyms     #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators       #-}

-- |
-- The Julia-set computation, shared verbatim by the cold and warm
-- executables so that the Accelerate program is identical in both.
--
-- Output: DIM x DIM x 4 array of Int32 (r,g,b,a per pixel); element (y,x,c)
-- sits at ((y*dim)+x)*4+c, identical to the CUDA kernel's
-- ptr[(x + y*dim)*4 + c] layout. Int32 matches CUDA's `int`.
module Julia (pixelsN) where

import qualified Data.Array.Accelerate as A
import           Data.Array.Accelerate (Acc, Array, DIM3, Exp, Scalar,
                                          Z (..), (:.) (..))
import           Data.Int              (Int32)

-- | Mirrors the CUDA __device__ function `julia`: returns 1 if the point
-- stays bounded for 200 iterations, 0 if it escapes (|z|^2 > 1000) at any
-- point during those 200 iterations.
--
-- Accelerate has no early `break` inside a device-side loop, so instead of
-- stopping the loop we track a `diverged` flag: once set, ar/ai are frozen,
-- which is observationally identical to the CUDA version's early return.
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

-- | The Accelerate program. `dim` is a Scalar Int input.
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
