module Data.Algorithm.SatSolver.Solver (
  -- * Type
  Assignment

  -- * Solve with literal selection
-- , solve
) where

  import qualified Data.Foldable                   as F
  import qualified Data.List                       as L
  import qualified Data.Map                        as M
  import qualified Data.Tuple                      as T

  import qualified Data.Algorithm.SatSolver.Fml    as Fml
  import qualified Data.Algorithm.SatSolver.Clause as Clause
  import qualified Data.Algorithm.SatSolver.Lit    as Lit
  import qualified Data.Algorithm.SatSolver.Var    as Var

  -- |'Assignement' type definition.
  type Assignment a = M.Map (Var.Var a) Bool

  -- |'reduceClause' @l@ @c@ reduces clause @c@ according to literal @l@.
  --
  -- >>> c = Clause.mk [Lit.mkPos' 1, Lit.mkNeg' 2, Lit.mkPos' 3]
  -- >>> Clause.reduce (Lit.mkPos' 1) c
  -- [+1,-2,+3]
  -- >>> Clause.reduce (Lit.mkNeg' 1) c
  -- [-2,+3]
  -- >>> Clause.reduce (Lit.mkPos' 4) c
  -- [+1,-2,+3]
  -- reduceClause :: (Eq a, Ord a) => Lit.Lit a -> Clause.Clause a -> Clause.Clause a
  -- To be implemented...

  -- |'reduce' @l@ @f@ reduces formula @f@ according to literal @l@.
  -- reduceFml :: (Eq a, Ord a) => Lit.Lit a -> Fml.Fml a -> Fml.Fml a
  -- To be implemented...

  -- Select a literal in a formula according to the following rules
  -- (order is relevant):
  --   1) select a literal from a unit clause, if any.
  --   2) otherwise select a literal that occurs only potive or negative, if any.
  --   2) select a most frequent literal, if any.
  --   3) return Nothing if the formula contains no literal.
  -- selectLit :: (Ord a) => Fml.Fml a -> Maybe (Lit.Lit a)
  -- To be implemented...

  -- |'solve' @f@ solves a CNF formula @f@. It returns an assignment
  -- if @f@ is satisfiable and @Nothing@ otherwise.
  -- solve :: (Ord a) => Fml.Fml a -> Maybe (Assignment a)
  -- To be implemented...
