module Data.Algorithm.SatSolver.Clause.Some
  ( -- * Testing
    clause1,
    clause2,
    clause3,
    clause4,
    clause5,
    clause6,
    clause7,
    clause8,
    clause9,
  )
where

import qualified Data.Algorithm.SatSolver.Clause as Clause
import qualified Data.Algorithm.SatSolver.Lit.Some as Lit.Some

clause1 :: Clause.Clause Int
clause1 = Clause.mk []

clause2 :: Clause.Clause Int
clause2 = Clause.mk [Lit.Some.posLit1]

clause3 :: Clause.Clause Int
clause3 = Clause.mk [Lit.Some.negLit1]

clause4 :: Clause.Clause Int
clause4 = Clause.mk [Lit.Some.posLit1, Lit.Some.negLit2, Lit.Some.posLit3]

clause5 :: Clause.Clause Int
clause5 = Clause.mk [Lit.Some.posLit1, Lit.Some.negLit2, Lit.Some.posLit3, Lit.Some.posLit1, Lit.Some.negLit2, Lit.Some.posLit3]

clause6 :: Clause.Clause Int
clause6 = Clause.mk [Lit.Some.negLit1, Lit.Some.negLit3, Lit.Some.negLit5, Lit.Some.negLit7]

clause7 :: Clause.Clause Int
clause7 = Clause.mk [Lit.Some.posLit2, Lit.Some.posLit4, Lit.Some.posLit9]

clause8 :: Clause.Clause Int
clause8 = Clause.mk [Lit.Some.negLit1, Lit.Some.posLit1, Lit.Some.posLit2]

clause9 :: Clause.Clause Int
clause9 = Clause.mk [Lit.Some.posLit1, Lit.Some.posLit1, Lit.Some.posLit1]
