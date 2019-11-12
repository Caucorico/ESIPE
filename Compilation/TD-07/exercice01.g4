grammar exercice01;

/* EOF */
s0: s EOF ;

/* s -> a s a | a s | b  */
s : ( ('a' s 'a'?) | 'b' ) ;

/* The other char than a and b cannot be recognized */
AB : ~([ab]) -> skip ;
