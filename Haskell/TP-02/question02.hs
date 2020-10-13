import Data.List

pairs :: [a] -> [(a, a)]
pairs [] = []
pairs (x:xs)
    | null xs = []
    | otherwise = (x, head xs) : pairs xs

evenElts :: [a] -> [a]
evenElts xs = [ xs !! a | a <- [0..length xs - 1], 0 == (a `mod` 2) ]

subLength :: [[a]] -> [([a], Int)]
subLength xs = [ (a, length (a)) | a <- xs ]

appOnPairs :: (a -> c) -> (b -> d) -> [(a, b)] -> [(c, d)]
appOnPairs f g xs = [ (f a, g b) | (a,b) <- xs]

factors :: (Eq a) => [a] -> [[a]]
