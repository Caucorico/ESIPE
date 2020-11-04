module Data.Algorithm.SatSolver.Lit
  ( -- * Type
    Lit (..),

    -- * Constructing
    mkNeg,
    mkPos,
    mkNeg',
    mkPos',
    fromString,
    neg,

    -- * Transforming
    getVar,
    toBool,
  )
where

import qualified Data.Algorithm.SatSolver.Utils as Utils
import qualified Data.Algorithm.SatSolver.Var as Var

-- | 'Lit' type
data Lit a = Neg (Var.Var a) | Pos (Var.Var a) deriving (Eq)

-- | Show instance
instance (Show a) => Show (Lit a) where
  show (Neg v) = '-' : show v
  show (Pos v) = '+' : show v

-- | Ord instance
instance (Ord a) => Ord (Lit a) where
  l `compare` l' = case getVar l `compare` getVar l' of
    LT -> LT
    EQ -> toBool l `compare` toBool l'
    GT -> GT

-- | 'mkNeg' @v@ makes a negative literal from propositional variable @v@.
--
--  >>> import qualified Data.Algorithm.SatSolver.Var as Var
--  >>> mkNeg (Var.mk "x1")
--  -"x1"
mkNeg :: Var.Var a -> Lit a
mkNeg = Neg

-- | 'mkPos' @v@ makes a positive literal from propositional variable @v@.
--
--  >>> import qualified Data.Algorithm.SatSolver.Var as Var
--  >>> mkPos (Var.mk "x1")
--  +"x1"
mkPos :: Var.Var a -> Lit a
mkPos = Pos

-- | 'mkNeg'' @n@ makes a propositional variable with name @n@ and next
--  makes a negative literal from this propositional variable.
--
--  >>> mkNeg' "x1"
--  -"x1"
mkNeg' :: a -> Lit a
mkNeg' = mkNeg . Var.mk

-- | 'mkPos'' @n@ makes a propositional variable with name @n@ and next
--  makes a positive literal from this propositional variable.
--
--  >>> mkPos' "x1"
--  +"x1"
mkPos' :: a -> Lit a
mkPos' = mkPos . Var.mk

-- | 'neg' @l@ returns the opposite literal of literal @l@.
--
--  >>> neg $ mkNeg' "x1"
--  +"x1"
--  >>> neg $ mkNeg' "x1"
--  -"x1"
neg (Neg v) = mkPos v
neg (Pos v) = mkNeg v

-- | 'getVar' @l@ return the propositional variable literal @l@ is defined on.
--
--  >>> getVar $ L.mkNeg' "x1"
--  "x1"
--  >>> getVar $ L.mkPos' "x1"
--  "x1"
getVar :: Lit a -> Var.Var a
getVar (Neg v) = v
getVar (Pos v) = v

-- | 'toBool' @l@ returns true if literal is positive and false otherwise.
--
--  >>> toBool  $ L.mkNeg' "x1"
--  False
--  >>> toBool  $ L.mkPos' "x1"
--  True
toBool :: Lit a -> Bool
toBool (Neg _) = False
toBool (Pos _) = True

-- | 'fromString' @s@ makes a literal from a string.
--  The first character must be either '-' or '+'
--  (leading and trailing whitespace are removed).
--
--  >>> fromString "-x1"
--  -"x1"
--  >>>  :type fromString "-x1"
--  fromString "-x1" :: Lit String
--  >>> fromString "+x1"
--  +"x1"
--  >>> :type fromString "+x1"
--  fromString "+x1" :: Lit String
fromString :: String -> Lit String
fromString = aux . Utils.trim
  where
    aux ('-' : n) = mkNeg' n
    aux ('+' : n) = mkPos' n
    aux l = error $ "Parse error for literal \"" ++ l ++ "\""
