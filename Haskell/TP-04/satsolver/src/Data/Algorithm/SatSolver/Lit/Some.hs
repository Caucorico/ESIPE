module Data.Algorithm.SatSolver.Lit.Some
  ( -- * Testing
    negLit1,
    posLit1,
    negLit2,
    posLit2,
    negLit3,
    posLit3,
    negLit4,
    posLit4,
    negLit5,
    posLit5,
    negLit6,
    posLit6,
    negLit7,
    posLit7,
    negLit8,
    posLit8,
    negLit9,
    posLit9,
    negLitEvtA,
    posLitEvtA,
    negLitEvtB,
    posLitEvtB,
    negLitEvtC,
    posLitEvtC,
  )
where

import qualified Data.Algorithm.SatSolver.Lit as Lit
import qualified Data.Algorithm.SatSolver.Var.Some as Var.Some

negLit1 :: Lit.Lit Int
negLit1 = Lit.mkNeg Var.Some.var1

posLit1 :: Lit.Lit Int
posLit1 = Lit.mkPos Var.Some.var1

negLit2 :: Lit.Lit Int
negLit2 = Lit.mkNeg Var.Some.var2

posLit2 :: Lit.Lit Int
posLit2 = Lit.mkPos Var.Some.var2

negLit3 :: Lit.Lit Int
negLit3 = Lit.mkNeg Var.Some.var3

posLit3 :: Lit.Lit Int
posLit3 = Lit.mkPos Var.Some.var3

negLit4 :: Lit.Lit Int
negLit4 = Lit.mkNeg Var.Some.var4

posLit4 :: Lit.Lit Int
posLit4 = Lit.mkPos Var.Some.var4

negLit5 :: Lit.Lit Int
negLit5 = Lit.mkNeg Var.Some.var5

posLit5 :: Lit.Lit Int
posLit5 = Lit.mkPos Var.Some.var5

negLit6 :: Lit.Lit Int
negLit6 = Lit.mkNeg Var.Some.var6

posLit6 :: Lit.Lit Int
posLit6 = Lit.mkPos Var.Some.var6

negLit7 :: Lit.Lit Int
negLit7 = Lit.mkNeg Var.Some.var7

posLit7 :: Lit.Lit Int
posLit7 = Lit.mkPos Var.Some.var7

negLit8 :: Lit.Lit Int
negLit8 = Lit.mkNeg Var.Some.var8

posLit8 :: Lit.Lit Int
posLit8 = Lit.mkPos Var.Some.var8

negLit9 :: Lit.Lit Int
negLit9 = Lit.mkNeg Var.Some.var9

posLit9 :: Lit.Lit Int
posLit9 = Lit.mkPos Var.Some.var9

negLitEvtA :: Lit.Lit String
negLitEvtA = Lit.mkNeg Var.Some.varEvtA

posLitEvtA :: Lit.Lit String
posLitEvtA = Lit.mkPos Var.Some.varEvtA

negLitEvtB :: Lit.Lit String
negLitEvtB = Lit.mkNeg Var.Some.varEvtB

posLitEvtB :: Lit.Lit String
posLitEvtB = Lit.mkPos Var.Some.varEvtB

negLitEvtC :: Lit.Lit String
negLitEvtC = Lit.mkNeg Var.Some.varEvtC

posLitEvtC :: Lit.Lit String
posLitEvtC = Lit.mkPos Var.Some.varEvtC
