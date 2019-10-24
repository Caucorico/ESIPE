
lexer grammar language_C;

/* EMPTY */
WHITE_SPACE : [ \t\r\n]+ -> skip ;

/* END LINE */
END_LINE    : ';' { System.out.println("END_LINE"); };

/*BLOCKS */
START_BLOCK : '{' { System.out.println("START BLOCK"); };
END_BLOCK   : '}' { System.out.println("END BLOCK"); };

/* CONTENT */
START_CONTENT : '(' { System.out.println("START CONTENT"); };
END_CONTENT   : ')' { System.out.println("END CONTENT"); };
CONTENT_SEPARATOR : ',' { System.out.println("CONTENT SEPARATOR"); };

/*NUMBER */
BINARY      : '0b'[0-1]+ { System.out.println("Binary"); };
/* OCTAL    : CANCELED  */
INT         : [0-9]+ { System.out.println("Integer"); };
LONG        : [0-9]+[Ll] { System.out.println("Long"); };
FLOAT       : [0-9]+[.]?[0-9]*[fF] { System.out.println("Float"); };
DOUBLE      : [0-9]+'.'[0-9]+ { System.out.println("Double"); };

/* OPERANDS : */
INCREMENT   : '++' { System.out.println("Incrementation"); };
DECREMENT   : '--' { System.out.println("Decrementation"); };
AFFECTATION : '=' { System.out.println("Affectation"); };
EQUAL       : '==' { System.out.println("Equal"); };
INFERIOR    : '<' {System.out.println("Inferieur"); };
SUPERIOR    : '>' { System.out.println("Superieur"); };

/* STRINGS */
STRING      : '"'.*?'"' { System.out.println("Chaine de caractere"); };
CHAR        : '\''.??'\'' { System.out.println("Caractere"); };

/* TYPES */
TYPES : (INT_TYPE|CHAR_TYPE|UNSIGNED_TYPE|POINTER_TYPE|FLOAT_TYPE|DOUBLE_TYPE) { System.out.println("Type"); };
INT_TYPE         : 'int' { System.out.println("Integer Type"); };
CHAR_TYPE        : 'char' { System.out.println("Char Type"); };
UNSIGNED_TYPE    : 'unsigned' { System.out.println("Unsigned Type"); };
POINTER_TYPE     : [*]+ { System.out.println("Pointer"); };
FLOAT_TYPE       : 'float' { System.out.println("Float"); };
DOUBLE_TYPE      : 'double' { System.out.println("Double"); };

/* OTHER : */
IDENTIFIER  : [a-zA-Z]+[a-zA-Z0-9]* { System.out.println("Identificateur"); };

