module Data.Algorithm.SatSolver.Clause
  ( -- * type
    Clause (..),

    -- * constructing
    mk,
    fromString,

    -- * properties

    -- isEmpty,
    -- isUnit,
    -- isMonotone,
    -- isNegMonotone,
    -- isPosMonotone,

    -- * querying

    -- , size
    -- , getVars
  )
where

import qualified Data.Algorithm.SatSolver.Lit as Lit
import qualified Data.Algorithm.SatSolver.Utils as Utils
import qualified Data.Algorithm.SatSolver.Var as Var
import qualified Data.Foldable as F
import qualified Data.List as L
import qualified Data.List.Split as L.Split

-- | 'Clause' type
newtype Clause a = Clause {getLits :: [Lit.Lit a]} deriving (Eq, Ord)

-- | Show instance
instance (Show a) => Show (Clause a) where
  show c = "[" ++ L.intercalate "," (L.map show ls) ++ "]"
    where
      ls = getLits c

-- | 'mk' @cs@ makes a clause from a list of literals.
--  Clauses are sorted and duplicate literals are removed.
--  Opposite literals are not removed.
--
--  >>> import qualified Data.Algorithm.SatSolver.Lit as Lit
--  >>> mk []
--  []
--  >>> mk [Lit.mkNeg' "x1"]
--  [-"x1"]
--  >>> mk [Lit.mkNeg' "x1", Lit.mkPos' "x2", Lit.mkNeg' "x3"]
--  [-"x1",+"x2",-"x3"]
--  >>> mk [Lit.mkNeg' "x1", Lit.mkPos' "x2", Lit.mkNeg' "x3", Lit.mkPos' "x1"]
--  [-"x1",+"x1",+"x2",-"x3"]
--  >>> mk [Lit.mkNeg' "x3", Lit.mkPos' "x2", Lit.mkNeg' "x1"]
--  [-"x1",+"x2",-"x3"]
--  >>> mk [Lit.mkPos' "x1", Lit.mkNeg' "x1"]
--  [-"x1",+"x1"]
mk :: (Ord a) => [Lit.Lit a] -> Clause a
mk = Clause . L.sort . L.nub

-- | 'isEmpty' @c@ returns true if clause @c@ contains no literal.
--
--  >>> import qualified Data.Algorithm.SatSolver.Lit as Lit
--  >>> let c = mk [] in isEmpty c
--  True
--  >>> let c = mk [Lit.mkNeg' "x1"] in isEmpty c
--  False
-- isEmpty :: Clause a -> Bool

-- isEmpty' (Clause lst) = null lst

-- | 'isUnit' @c@ returns true iff clause @c@ contains exactly one literal.
--
--  >>> import qualified Data.Algorithm.SatSolver.Lit as Lit
--  >>> let c = mk [] in isUnit c
--  False
--  >>> let c = mk [Lit.mkNeg' "x1"] in isUnit c
--  True
--  >>> let c = mk [Lit.mkNeg' "x1", Lit.mkPos' "x2"] in isUnit c
--  False
-- isUnit :: Clause a -> Bool

-- | 'isMonotone' @c@ returns true iff clause @c@ is monotone.
--  A clause is monotone if all literals are either positive
--  or negative.
--  An empty clause is monotone.
--
--  >>> import qualified Data.Algorithm.SatSolver.Lit as Lit
--  >>> isMonotone $ mk []
--  True
--  >>> isMonotone $ mk [Lit.mkPos' "x1", Lit.mkPos' "x2", Lit.mkPos' "x3"]
--  True
--  >>> isMonotone $ mk [Lit.mkNeg' "x1", Lit.mkPos' "x2", Lit.mkNeg' "x3"]
--  False
--  >>> isMonotone $ mk [Lit.mkNeg' "x1", Lit.mkNeg' "x2", Lit.mkNeg' "x3"]
--  True
-- isMonotone :: Clause a -> Bool

-- | 'isNegMonotone' @c@ returns true iff clause @c@ is negative monotone.
--  A clause is negative monotone if all literals are negative.
--  An empty clause is negative monotone.
--
--  >>> import qualified Data.Algorithm.SatSolver.Lit as Lit
--  >>> isNegMonotone $ mk []
--  True
--  >>> isNegMonotone $ mk [Lit.mkPos' "x1", Lit.mkPos' "x2", Lit.mkPos' "x3"]
--  False
--  >>> isNegMonotone $ mk [Lit.mkNeg' "x1", Lit.mkPos' "x2", Lit.mkNeg' "x3"]
--  False
--  >>> isNegMonotone $ mk [Lit.mkNeg' "x1", Lit.mkNeg' "x2", Lit.mkNeg' "x3"]
--  True
--  isNegMonotone :: Clause a -> Bool
--  To be implemented...

-- | 'isPosMonotone' @c@ returns true iff clause @c@ is positive monotone.
--  A clause is negative monotone if all literals are positive.
--  An empty clause is positive monotone.
--
--  >>> import qualified Data.Algorithm.SatSolver.Lit as Lit
--  >>> isPosMonotone $ mk []
--  True
--  >>> isPosMonotone $ mk [Lit.mkPos' "x1", Lit.mkPos' "x2", Lit.mkPos' "x3"]
--  True
--  >>> isPosMonotone $ mk [Lit.mkNeg' "x1", Lit.mkPos' "x2", Lit.mkNeg' "x3"]
--  False
--  >>> isPosMonotone $ mk [Lit.mkNeg' "x1", Lit.mkNeg' "x2", Lit.mkNeg' "x3"]
--  False
--  isPosMonotone :: Clause a -> Bool
--  To be implemented...

-- | 'size' @c@ returns the number of literals in clause @c@.
--
--  >>> import qualified Data.Algorithm.SatSolver.Lit as Lit
--  >>> let c mk [] in size c
--  0
--  >>> let c = mk [Lit.mkPos' "x1"] in size c
--  1
--  >>> let c = mk [Lit.mkPos' "x1", Lit.mkPos' "x1"] in size c
--  1
--  >>> let c = mk [Lit.mkPos' i | i <- [1..100]] in size c
--  100
--  size :: Clause a -> Int
--  To be implemented...

-- | 'getVars' @c@ returns the distinct propositional variables that
--  occur in clause @c@.
--
--  >>> import qualified Data.Algorithm.SatSolver.Lit as Lit
--  >>>  let c = mk [] in getVars c
--  []
--  >>> let c = mk [Lit.mkPos' "x1"] in getVars c
--  ["x1"]
--  >>>  let c = mk [Lit.mkPos' "x1", Lit.mkNeg' "x2", Lit.mkPos' "x3"] in getVars c
--  ["x1","x2","x3"]
--  >>> let c = mk [Lit.mkPos' "x1", Lit.mkNeg' "x2", Lit.mkNeg' "x1"] in getVars c
--  ["x1","x2"]
--  getVars :: (Eq a) => Clause a -> [Var.Var a]
--  To be implemented...

-- | 'fromString' @c@ makes a clause from a string.
--  The first character must be '[' and the last character must be ']'
--  (leading and trailing whitespace are removed).
--
--  >>> fromString "[]"
--  []
--  >>> :type Clause.fromString "[]"
--  Clause.fromString "[]" :: Clause.Clause String
--  >>> Clause.fromString "[+x1, -x3, +x2, -x4]"
--  [+"x1",+"x2",-"x3",-"x4"]
--  >>> :type Clause.fromString "[+x1, -x3, +x2, -x4]"
--  Clause.fromString "[+x1, -x3, +x2, -x4]" :: Clause.Clause String
fromString :: String -> Clause String
fromString = aux . Utils.trim
  where
    aux [] = error $ "Literal parse error for clause \"" ++ [] ++ "\""
    aux s =
      if L.head s == '[' && L.last s == ']'
        then aux' . L.init $ L.tail s
        else error $ "Literal parse error for clause \"" ++ s ++ "\""
    aux' [] = mk []
    aux' s = mk . L.map Lit.fromString $ L.Split.splitOn "," s
