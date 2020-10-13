import Data.List

mirror :: Eq a => [a] -> [a] -> Bool
{- Methode normale : -}
{- mirror xs ys = xs == reverse ys -}

{- Methode recursive -}
mirror xs ys
    | null xs && null ys = True
    | not (null xs) && not (null ys) = head xs == last ys && mirror (tail xs) (init ys)
    | otherwise = False


{- Quicksort -}
quicksort :: Ord a => [a] -> [a]
quicksort [] = []
quicksort (x:xs) = (quicksort firstpart) ++ [x] ++ (quicksort lastpart) where
    firstpart = filter ( < x ) xs
    lastpart  = filter ( >= x ) xs 


{- Permute O(n²) -}
permute :: Eq a => [a] -> [a] -> Bool
permute [] [] = True
permute [] _ = False
permute (x:xs) ys = x `elem` ys && permute xs ys' where
    ys' = delete x ys

{- Permute O(nlog n) -}
permute' :: Ord a => [a] -> [a] -> Bool
permute' xs ys = quicksort xs == quicksort ys
