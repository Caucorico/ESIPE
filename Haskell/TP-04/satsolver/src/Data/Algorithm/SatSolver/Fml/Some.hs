module Data.Algorithm.SatSolver.Fml.Some (
  -- * Testing Var Int
  fml1
, fml2
, fml3
, fml4
, fml5
, fml6
, fml7
, fml8
, fml9
, fml10
, fml11

  -- * Teting Var String
, fmlEvt1
, fmlEvt2
) where

  import qualified Data.Algorithm.SatSolver.Clause   as Clause
  import qualified Data.Algorithm.SatSolver.Fml      as Fml
  import qualified Data.Algorithm.SatSolver.Lit.Some as Lit.Some

  -- |Satisfiable empty CNF formula.
  --
  -- >>> fml1
  -- []
  fml1 :: Fml.Fml Int
  fml1 = Fml.mk []

  -- |Satisfiable CNF formula.
  -- \[(x_1 \vee \neg x_2 \vee \neg x_3 \vee x_4)\]
  --
  -- >>> fml2
  -- [[+1,-2,-3,+4]]
  fml2 :: Fml.Fml Int
  fml2 = Fml.mk [c1]
    where
      c1 = Clause.mk [Lit.Some.posLit1, Lit.Some.negLit2, Lit.Some.negLit3, Lit.Some.posLit4]

  -- |Satisfiable CNF formula.
  -- \[(x_1 \vee \neg x_3) \wedge (x_1 \vee x_4 \vee \neg x_5) \wedge (x_2 \vee x_3 \vee x_4) \wedge (\neg x_1 \vee \neg x_3 \vee \neg x_5) \wedge (x_4 \vee x_5) \wedge (\neg x_3 \vee \neg x_4 \vee x_5) \wedge (x_2)\]
  --
  -- >>> fml3
  -- [[+1,-3],[+1,+4,-5],[+2,+3,+4],[-1,-3,-5],[+4,+5],[-3,-4,+5],[+2]]
  fml3 :: Fml.Fml Int
  fml3 = Fml.mk [c1, c2, c3, c4, c5, c6, c7]
    where
      c1 = Clause.mk [Lit.Some.posLit1, Lit.Some.negLit3]
      c2 = Clause.mk [Lit.Some.posLit1, Lit.Some.posLit4, Lit.Some.negLit5]
      c3 = Clause.mk [Lit.Some.posLit2, Lit.Some.posLit3, Lit.Some.posLit4]
      c4 = Clause.mk [Lit.Some.negLit1, Lit.Some.negLit3, Lit.Some.negLit5]
      c5 = Clause.mk [Lit.Some.posLit4, Lit.Some.posLit5]
      c6 = Clause.mk [Lit.Some.negLit3, Lit.Some.negLit4, Lit.Some.posLit5]
      c7 = Clause.mk [Lit.Some.posLit2]

  -- |Satisfiable CNF formula.
  -- \[(x_1 \vee \neg x_1 \vee \neg x_3) \wedge (x_1 \vee x_4 \vee \neg x_5 \vee x_6 \vee \neg x_7) \wedge (x_2 \vee x_3 \vee x_4 \vee x_8 \vee x_9)\]
  --
  -- >>> fml4
  -- [[+1,-1,-3],[+1,+4,-5,+6,-7],[+2,+3,+4,+8,+9]]
  fml4 :: Fml.Fml Int
  fml4 = Fml.mk [c1, c2, c3]
    where
      c1  = Clause.mk [Lit.Some.posLit1, Lit.Some.negLit3, Lit.Some.negLit1]
      c2  = Clause.mk [Lit.Some.posLit1, Lit.Some.posLit4, Lit.Some.negLit5, Lit.Some.posLit6, Lit.Some.negLit7]
      c3  = Clause.mk [Lit.Some.posLit2, Lit.Some.posLit3, Lit.Some.posLit4, Lit.Some.posLit8, Lit.Some.posLit9]

  -- |Satisfiable CNF formula.
  -- \[(x_1 \vee \neg x_2) \wedge (x_1 \vee x_4) \wedge (x_8 \vee x_9) \wedge (\neg x_1 \vee \neg x_8) \wedge (x_4 \vee x_5)\]
  --
  -- >>> fml5
  -- [[+1,-2],[+1,+4],[+8,+9],[-1,-8],[+4,+5]]
  fml5 = Fml.mk [c1, c2, c3, c4, c5]
    where
      c1  = Clause.mk [Lit.Some.posLit1, Lit.Some.negLit2]
      c2  = Clause.mk [Lit.Some.posLit1, Lit.Some.posLit4]
      c3  = Clause.mk [Lit.Some.posLit8, Lit.Some.posLit9]
      c4  = Clause.mk [Lit.Some.negLit1, Lit.Some.negLit8]
      c5  = Clause.mk [Lit.Some.posLit4, Lit.Some.posLit5]

  -- |Unsatisfiable CNF formula.
  -- \[(x_1 \vee x_2) \wedge (\neg x_1 \vee x_2) \wedge (x_1 \vee \neg x_2) \wedge (\neg x_1 \vee \neg x_2)\]
  --
  -- >>> fml6
  -- [[+1,+2],[-1,+2],[+1,-2],[-1,-2]]
  fml6 :: Fml.Fml Int
  fml6 = Fml.mk [c1, c2, c3, c4]
    where
      c1 = Clause.mk [Lit.Some.posLit1, Lit.Some.posLit2]
      c2 = Clause.mk [Lit.Some.negLit1, Lit.Some.posLit2]
      c3 = Clause.mk [Lit.Some.posLit1, Lit.Some.negLit2]
      c4 = Clause.mk [Lit.Some.negLit1, Lit.Some.negLit2]

  -- |Satisfiable CNF formula
  -- \[(x_1) \wedge (\neg x_2 ) \wedge (x_3) \wedge (x_4) \wedge (\neg x_5) \wedge (x_6) \wedge (\neg x_7) \wedge (\neg x_8) \wedge (\neg x_9)\]
  --
  -- >>> fml7
  -- [[+1],[-2],[+3],[+4],[-5],[+6],[-7],[-8],[-9]]
  fml7 :: Fml.Fml Int
  fml7 = Fml.mk [c1, c2, c3, c4, c5, c6, c7, c8, c9]
    where
      c1 = Clause.mk [Lit.Some.posLit1]
      c2 = Clause.mk [Lit.Some.negLit2]
      c3 = Clause.mk [Lit.Some.posLit3]
      c4 = Clause.mk [Lit.Some.posLit4]
      c5 = Clause.mk [Lit.Some.negLit5]
      c6 = Clause.mk [Lit.Some.posLit6]
      c7 = Clause.mk [Lit.Some.negLit7]
      c8 = Clause.mk [Lit.Some.negLit8]
      c9 = Clause.mk [Lit.Some.negLit9]

  -- |Satisfiable CNF formula
  -- \[(x_1 \vee x_2) \wedge (x_2 \vee \neg x_3) \wedge (\neg x_3 \vee x_4) \wedge (x_4 \vee \neg x_5) \wedge (\neg x_5 \vee x_6) \wedge (x_6 \vee \neg x_7) \wedge (\neg x_7 \vee x_8) \wedge (x_8 \vee \neg x_9) \wedge (\neg x_9 \vee \neg x_1)\]
  --
  -- >>> fml8
  -- [[+1,+2],[+2,-3],[-3,+4],[+4,-5],[-5,+6],[+6,-7],[-7,+8],[+8,-9],[-1,-9]]
  fml8 :: Fml.Fml Int
  fml8 = Fml.mk [c1, c2, c3, c4, c5, c6, c7, c8, c9]
    where
      c1 = Clause.mk [Lit.Some.posLit1, Lit.Some.posLit2]
      c2 = Clause.mk [Lit.Some.posLit2, Lit.Some.negLit3]
      c3 = Clause.mk [Lit.Some.negLit3, Lit.Some.posLit4]
      c4 = Clause.mk [Lit.Some.posLit4, Lit.Some.negLit5]
      c5 = Clause.mk [Lit.Some.negLit5, Lit.Some.posLit6]
      c6 = Clause.mk [Lit.Some.posLit6, Lit.Some.negLit7]
      c7 = Clause.mk [Lit.Some.negLit7, Lit.Some.posLit8]
      c8 = Clause.mk [Lit.Some.posLit8, Lit.Some.negLit9]
      c9 = Clause.mk [Lit.Some.negLit9, Lit.Some.negLit1]

  -- |Satisfiable CNF formula
  -- \[(x_1) \wedge (\neg x_2) \wedge (x_3) \wedge (\neg x_1 \vee x_2 \vee \neg x_4) \wedge (\neg x_3 \vee \neg x_5) \wedge (x_4 \vee x_5 \vee \neg x_6) \wedge (x_6 \vee x_7) \wedge (\neg x_7 \vee x_8) \wedge (\neg x_8 \vee x_9)\]
  --
  -- >>> fml9
  -- [[+1],[-2],[+3],[-1,+2,-4],[-3,-5],[+4,+5,-6],[+6,+7],[-7,+8],[-8,+9]]
  fml9 :: Fml.Fml Int
  fml9 = Fml.mk [c1, c2, c3, c4, c5, c6, c7, c8, c9]
    where
      c1 = Clause.mk [Lit.Some.posLit1]
      c2 = Clause.mk [Lit.Some.negLit2]
      c3 = Clause.mk [Lit.Some.posLit3]
      c4 = Clause.mk [Lit.Some.negLit1, Lit.Some.posLit2, Lit.Some.negLit4]
      c5 = Clause.mk [Lit.Some.negLit3, Lit.Some.negLit5]
      c6 = Clause.mk [Lit.Some.posLit4, Lit.Some.posLit5, Lit.Some.negLit6]
      c7 = Clause.mk [Lit.Some.posLit6, Lit.Some.posLit7]
      c8 = Clause.mk [Lit.Some.negLit7, Lit.Some.posLit8]
      c9 = Clause.mk [Lit.Some.negLit8, Lit.Some.posLit9]

  -- |Unsatisfiable CNF formula
  -- \[(x_1) \wedge (x_2) \wedge (x_3) \wedge (\neg x_1 \vee x_4) \wedge (\neg x_2 \vee x_5) \wedge (\neg x_3 \vee x_6) \wedge (\neg x_4 \vee x_7) \wedge (\neg x_5 \vee x_8) \wedge (\neg x_7 \vee \neg x_8)\]
  --
  -- >>> fml10
  -- [[+1],[+2],[+3],[-1,+4],[-2,+5],[-3,+6],[-4,+7],[-5,+8],[-7,-8]]
  fml10 :: Fml.Fml Int
  fml10 = Fml.mk [c1, c2, c3, c4, c5, c6, c7, c8, c9]
    where
      c1 = Clause.mk [Lit.Some.posLit1]
      c2 = Clause.mk [Lit.Some.posLit2]
      c3 = Clause.mk [Lit.Some.posLit3]
      c4 = Clause.mk [Lit.Some.negLit1, Lit.Some.posLit4]
      c5 = Clause.mk [Lit.Some.negLit2, Lit.Some.posLit5]
      c6 = Clause.mk [Lit.Some.negLit3, Lit.Some.posLit6]
      c7 = Clause.mk [Lit.Some.negLit4, Lit.Some.posLit7]
      c8 = Clause.mk [Lit.Some.negLit5, Lit.Some.posLit8]
      c9 = Clause.mk [Lit.Some.negLit7, Lit.Some.negLit8]

  -- |Satisfiable CNF formula
  -- \[(x_1 \vee x_2) \wedge (\neg x_2 \vee \neg x_3) \wedge (x_3 \vee x_4) \wedge (\neg x_4 \vee \neg x_5) \wedge (x_5 \vee x_6) \wedge (\neg x_6 \vee \neg x_7) \wedge (x_7 \vee x_8) \wedge (\neg x_8 \vee \neg x_9) \wedge (x_9 \vee \neg x_1)\]
  --
  -- >>> fml11
  -- [[+1,+2],[-2,-3],[+3,+4],[-4,-5],[+5,+6],[-6,-7],[+7,+8],[-8,-9],[-1,+9]]
  fml11 :: Fml.Fml Int
  fml11 = Fml.mk [c1, c2, c3, c4, c5, c6, c7, c8, c9]
    where
      c1 = Clause.mk [Lit.Some.posLit1, Lit.Some.posLit2]
      c2 = Clause.mk [Lit.Some.negLit2, Lit.Some.negLit3]
      c3 = Clause.mk [Lit.Some.posLit3, Lit.Some.posLit4]
      c4 = Clause.mk [Lit.Some.negLit4, Lit.Some.negLit5]
      c5 = Clause.mk [Lit.Some.posLit5, Lit.Some.posLit6]
      c6 = Clause.mk [Lit.Some.negLit6, Lit.Some.negLit7]
      c7 = Clause.mk [Lit.Some.posLit7, Lit.Some.posLit8]
      c8 = Clause.mk [Lit.Some.negLit8, Lit.Some.negLit9]
      c9 = Clause.mk [Lit.Some.posLit9, Lit.Some.negLit1]

  -- |Satisfiable CNF formula
  -- \[(\texttt{evt A} \vee \texttt{evt B}) \wedge (\neg \texttt{evt A} \vee \texttt{evt C}) \wedge (\texttt{evt A} \vee \neg \texttt{evt B} \neg \texttt{evt C})\]
  --
  -- >>> fmlEvt2
  -- [[+"Evt A"],[-"Evt A",+"Evt B"],[-"Evt B",-"Evt C"],[-"Evt A",-"Evt B",+"Evt C"]]
  fmlEvt1 :: Fml.Fml String
  fmlEvt1 = Fml.mk [c1, c2, c3]
    where
      c1 = Clause.mk [Lit.Some.posLitEvtA, Lit.Some.posLitEvtB]
      c2 = Clause.mk [Lit.Some.negLitEvtA, Lit.Some.posLitEvtC]
      c3 = Clause.mk [Lit.Some.posLitEvtA, Lit.Some.negLitEvtB, Lit.Some.negLitEvtC]

  -- |Unsatisfiable CNF formula
  -- \[(\texttt{evt A}) \wedge (\neg \texttt{evt A} \vee \texttt{evt B}) \wedge (\neg \texttt{evt B} \vee \neg \texttt{evt C}) \wedge (\texttt{evt A} \vee \neg \texttt{evt B} \vee \texttt{evt C})\]
  --
  -- >>> fmlEvt2
  -- [[+"Evt A"],[-"Evt A",+"Evt B"],[-"Evt B",-"Evt C"],[-"Evt A",-"Evt B",+"Evt C"]]
  fmlEvt2 :: Fml.Fml String
  fmlEvt2 = Fml.mk [c1, c2, c3, c4]
    where
      c1 = Clause.mk [Lit.Some.posLitEvtA]
      c2 = Clause.mk [Lit.Some.negLitEvtA, Lit.Some.posLitEvtB]
      c3 = Clause.mk [Lit.Some.negLitEvtB, Lit.Some.negLitEvtC]
      c4 = Clause.mk [Lit.Some.negLitEvtA, Lit.Some.negLitEvtB, Lit.Some.posLitEvtC]
