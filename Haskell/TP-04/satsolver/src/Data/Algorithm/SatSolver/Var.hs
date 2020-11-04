module Data.Algorithm.SatSolver.Var
  ( -- * Type
    Var (..),

    -- * Constructing
    mk,
  )
where

-- | 'Var' type
newtype Var a = Var {getName :: a} deriving (Eq, Ord)

-- | Show instance
instance (Show a) => Show (Var a) where
  show = show . getName

-- | 'mk' @n@ makes a propositional variable with name @n@.
--
--  >>> [mk i | i <- [1..4]]
--  [1,2,3,4]
--  >>> [mk i | i <- ['a'..'d']]
--  ['a','b','c','d']
--  >>> [mk o | o <- [LT .. GT]]
--  [LT,EQ,GT]
mk n = Var {getName = n}
