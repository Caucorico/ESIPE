/** Extract some French and English words *
  */
lexer grammar td3_ex1;
FRENCH : ('de' | 'à' | 'le' | 'la' | 'et' | 'il' | 'les' | 'un' | 'en' | 'du')
     { System.out.println("French ");  }
     ;
ENGLISH : ('the' | 'of' | 'and' | 'to' | 'a' | 'his' | 'in' | 'with' | 'I' | 'which')
     { System.out.println("English ");  }
     ;
OTHER: [\P{Alphabetic}]+ -> skip ;
CLEAR : [\p{Alphabetic}]+  -> skip ; 
SKIP_IN : ([\P{Alphabetic}]?'in'[\P{Alphabetic}]?) ;
