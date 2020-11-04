module Data.Algorithm.SatSolver.Fml (
  -- * type
  Fml(..)

  -- * constructing
, mk
, (/++/)

  -- * testing
-- , isEmpty
-- , isSatisfied
-- , hasUnsatisfiedClause

  -- * querying
-- , getLits
-- , getVars
-- , getUnitClauses
-- , size
-- , selectMostFrequentLit
-- , selectMonotoneLit

  -- * Transforming
-- , toNormal
-- , toDIMACSString
)
where

  import qualified Data.Foldable                   as F
  import qualified Data.List                       as L
  import qualified Data.Map.Strict                 as M
  import           Data.Maybe
  import qualified Data.Set                        as S
  import qualified Data.Tuple                      as T

  import qualified Data.Algorithm.SatSolver.Clause as Clause
  import qualified Data.Algorithm.SatSolver.Lit    as Lit
  import qualified Data.Algorithm.SatSolver.Utils  as Utils
  import qualified Data.Algorithm.SatSolver.Var    as Var

  -- | 'Fml' type
  newtype Fml a = Fml { getClauses :: [Clause.Clause a] }

  -- | show instance
  instance (Show a) => Show (Fml a) where
    show fml = "[" ++ L.intercalate "," (L.map show cs) ++ "]"
      where
        cs = getClauses fml

  -- |'mk' @cs@ makes a forumula from a list of clauses.
  -- Duplicate clauses are removed.
  --
  -- >>> import qualified Data.Algoritm.SatSolver.Clause as Clause
  -- >>> import qualified Data.Algoritm.SatSolver.Lit as Lit
  -- >>> mk []
  -- []
  -- >>> c1 = Clause.mk [Lit.mkPos' "x1", Lit.mkNeg' "x2", Lit.mkNeg' "x3"]
  -- >>> c2 = Clause.mk [Lit.mkNeg' "x1", Lit.mkNeg' "x2"]
  -- >>> c3 = Clause.mk [Lit.mkPos' "x1", Lit.mkPos' "x2", Lit.mkPos' "x3"]
  -- >>> mk [c1, c2, c3]
  -- [[-"x1",-"x2"],[+"x1",-"x2",-"x3"],[+"x1",+"x2",+"x3"]]
  mk :: (Ord a) => [Clause.Clause a] -> Fml a
  mk = Fml . L.nub . L.sort

  -- |'isEmpty' @f@ returns true if formula @f@ contains no clause.
  --
  -- >>> let f = mk [] in isEmpty f
  -- True
  -- isEmpty :: Fml a -> Bool
  -- To be implemented...

  -- |'isSatisfied' @f@ returns true if forumla @f@ is satisfied.
  -- This reduces to testing if @f@ contains no clause.
  --
  -- >>> let f = mk [] in isSatisfied f
  -- True
  -- isSatisfied :: Fml a -> Bool
  -- To be implemented...

  -- |'getLits' @f@ returns all literals of formula @f@.
  -- Duplicate are not removed.
  --
  -- >>> let f = mk [] in getLits f
  -- []
  -- >>> f
  -- [[+"x1",-"x2",-"x3"],[-"x1",-"x2"],[+"x1",+"x2",+"x3"]]
  -- >>> getLits f
  -- [+"x1",-"x2",-"x3",-"x1",-"x2",+"x1",+"x2",+"x3"]
  -- getLits :: Fml a -> [Lit.Lit a]
  -- To be implemented...

  -- |'getVars' @f@ returns the distinct propositional variables of
  -- formula @f@.
  --
  -- >>> let f = mk [] in getVars f
  -- []
  -- >>> f
  -- [[-"x1",-"x2"],[+"x1",-"x2",-"x3"],[+"x1",+"x2",+"x3"]]
  -- >>> getVars f
  -- ["x1","x2","x3"]
  -- getVars :: (Ord a) => Fml a -> [Var.Var a]
  -- To be implemented...

  -- |'getUnitClauses' @f@ returns the unit clauses of formula @f@.
  --
  -- >>> :type f
  -- f :: Fml [Char]
  -- >>> f
  -- [[+"x1"],[-"x1",+"x2"],[-"x2"]]
  -- >>> getUnitClauses f
  -- [[+"x1"],[-"x2"]]
  -- getUnitClauses :: (Ord a) => Fml a -> [Clause.Clause a]
  -- To be implemented...

  -- |'selectMostFrequentLit' @f@ returns the most frequent variable of formula @f@.
  -- The number of occurrences of a variable is the number of occurrences of the
  -- negative literals plus the number of occurrences of the positive literal.
  --
  -- >>> :type f
  -- f :: Fml [Char]
  -- >>> f
  -- [[-"x4",+"x5"],[+"x1",+"x3",-"x4"],[-"x1",+"x5"],[+"x1",+"x2"],[+"x1",-"x3"]]
  -- >>> Fml.selectMostFrequentLit f
  -- Just +"x1"
  -- selectMostFrequentLit :: (Ord a) => Fml a -> Maybe (Lit.Lit a)
  -- To be implemented...

  -- |'selectMonotoneLit' @f@ returns (if any) a literal that occurs only negatively or
  -- only positively in formula @f@.
  --
  -- >>> selectMonotoneLit $ mk []
  -- Nothing
  -- >>> f
  -- [[+1,+2],[-1,+3],[+1,+3],[-3,+4]]
  -- >>> selectMonotoneLit f
  -- Just +2
  -- >>> f'
  -- [[+1,+2],[-1,+3],[+1,+3],[-3,+4],[-2,-4]]
  -- >>> selectMonotoneLit f'
  -- Nothing
  -- selectMonotoneLit :: (Ord a) => Fml a -> Maybe (Lit.Lit a)
  -- To be implemented...

  -- |'hasUnsatisfiedClause' @f@ returns true if formula @f@ contains
  -- at least one empty clause.
  --
  -- >>> :type f
  -- f :: Fml [Char]
  -- >>> f
  -- [[+"x1",-"x2",-"x3"],[],[-"x1",-"x2"],[+"x1",+"x2",+"x3"]]
  -- >>>  hasUnsatisfiedClause f
  -- True
  -- >>> :type f'
  -- f' :: Fml [Char]
  -- >>> f'
  -- [[+"x1",-"x2",-"x3"],[-"x1",-"x2"],[+"x1",+"x2",+"x3"]]
  -- >>> hasUnsatisfiedClause f'
  -- False
  -- hasUnsatisfiedClause :: Fml a -> Bool
  -- To be implemented...

  -- |'size' @f@ returns the number of clauses (including empty clauses)
  -- in forumla @f@.
  --
  -- >>> :type f
  -- f :: Fml [Char]
  -- >>> f
  -- [[-"x4",+"x5"],[+"x1",+"x3",-"x4"],[-"x1",+"x5"],[+"x1",+"x2"],[+"x1",-"x3"]]
  -- >>> size f
  -- 5
  -- size :: Fml a -> Int
  -- To be implemented...

  -- |Union of two formulae.
  --
  -- >>> :type f
  -- f :: Fml [Char]
  -- >>> f
  -- [[+"x1",-"x2",-"x3"],[-"x1",-"x2"],[+"x1",+"x2",+"x3"]]
  -- >>> :type f'
  -- f' :: Fml [Char]
  -- >>> f'
  -- [[+"x1"],[+"x3",-"x3"],[-"x1",+"x3"],[-"x2",+"x4"]]
  -- f /++/ f'
  -- [[+"x1",-"x2",-"x3"],[-"x1",-"x2"],[+"x1",+"x2",+"x3"],[+"x1"],[+"x3",-"x3"],[-"x1",+"x3"],[-"x2",+"x4"]]
  (/++/) :: (Ord a) => Fml a -> Fml a -> Fml a
  f /++/ f' = mk $ getClauses f ++ getClauses f'

  -- |'toNormal' @f@ transforms a CNF formula into normal form.
  --
  -- >>> :type f
  -- f :: Fml [Char]
  -- >>> f
  -- [[-"x4",+"x5"],[+"x1",+"x3",-"x4"],[-"x1",+"x5"],[+"x1",+"x2"],[+"x1",-"x3"]]
  -- >>> toNormal f
  -- [[-4,+5],[+1,+3,-4],[-1,+5],[+1,+2],[+1,-3]]
  -- >>> :type f'
  -- f' :: Fml String
  -- >>> f'
  -- [[+"Evt A"],[-"Evt A",+"Evt B"],[-"Evt B",-"Evt C"],[-"Evt A",-"Evt B",+"Evt C"]]
  -- >>> toNormal f'
  -- [[+1],[-1,+2],[-2,-3],[-1,-2,+3]]
  -- toNormal :: (Ord a, Ord b ,Num b, Enum b) => Fml a -> Fml b
  -- To be implemented...

  -- |'toDIMACSString' transforms CNF formula @f@ into a DIMACS string.
  --
  -- >>> :type f
  -- f :: Fml [Char]
  -- >>> f
  -- [[-"x4",+"x5"],[+"x1",+"x3",-"x4"],[-"x1",+"x5"],[+"x1",+"x2"],[+"x1",-"x3"]]
  -- >>> putStr $ toDIMACSString f
  -- c haskell rule
  -- p cnf 9 9
  -- +1 +2 0
  -- +2 -3 0
  -- -3 +4 0
  -- +4 -5 0
  -- -5 +6 0
  -- +6 -7 0
  -- -7 +8 0
  -- +8 -9 0
  -- -1 -9 0
  -- >>> f'
  -- [[+"Evt A",+"Evt B"],[-"Evt A",+"Evt C"],[+"Evt A",-"Evt B",-"Evt C"]]
  -- >>> putStr $ toDIMACSString f'
  -- c haskell rule
  -- p cnf 3 3
  -- +1 +2 0
  -- -1 +3 0
  -- +1 -2 -3 0
  -- toDIMACSString :: (Ord a) => Fml a -> String
  -- To be implemented...
