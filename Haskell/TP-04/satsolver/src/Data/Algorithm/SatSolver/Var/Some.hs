module Data.Algorithm.SatSolver.Var.Some
  ( -- * Testing Var Int
    var1,
    var2,
    var3,
    var4,
    var5,
    var6,
    var7,
    var8,
    var9,

    -- * Testing Var String
    varEvtA,
    varEvtB,
    varEvtC,
  )
where

import qualified Data.Algorithm.SatSolver.Var as Var

var1 :: Var.Var Int
var1 = Var.mk 1

var2 :: Var.Var Int
var2 = Var.mk 2

var3 :: Var.Var Int
var3 = Var.mk 3

var4 :: Var.Var Int
var4 = Var.mk 4

var5 :: Var.Var Int
var5 = Var.mk 5

var6 :: Var.Var Int
var6 = Var.mk 6

var7 :: Var.Var Int
var7 = Var.mk 7

var8 :: Var.Var Int
var8 = Var.mk 8

var9 :: Var.Var Int
var9 = Var.mk 9

varEvtA :: Var.Var String
varEvtA = Var.mk "Evt A"

varEvtB :: Var.Var String
varEvtB = Var.mk "Evt B"

varEvtC :: Var.Var String
varEvtC = Var.mk "Evt C"
