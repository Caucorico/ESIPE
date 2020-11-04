module Data.Algorithm.SatSolver.Utils (
  -- * List
  safeHead
, safeLast

  -- * String
,  trim
) where

  import qualified Data.Char as C
  import qualified Data.List as L

  -- |'safeHead' @xs@ return the first element of list @xs@ if it is
  -- not empty and @Nothing@ otherwise.
  --
  -- >>> safeHead []
  -- Nothing
  -- >>> safeHead [1]
  -- Just 1
  -- >>> safeHead [1..5]
  -- Just 1
  safeHead :: [a] -> Maybe a
  safeHead []      = Nothing
  safeHead (x : _) = Just x

  -- |'safeHead' @xs@ return the last element of list @xs@ if it is
  -- not empty and @Nothing@ otherwise.
  --
  -- >>>  safeLast []
  -- Nothing
  -- >>> safeLast [1]
  -- Just 1
  -- >>> safeLast [1..5]
  -- Just 5
  safeLast :: [a] -> Maybe a
  safeLast [] = Nothing
  safeLast xs = Just $ L.last xs

  -- |'trim' @xs@ deletes the leading and trailing whitespace of @xs@.
  --
  -- >>> trim "  ab c de  "
  -- "ab c de"
  -- >>> trim "  ab c de"
  -- "ab c de"
  -- >>> trim "ab c de  "
  -- "ab c de"
  -- >> trim "  "
  -- ""
  trim :: String -> String
  trim = f . f
     where
       f = L.reverse . L.dropWhile C.isSpace
