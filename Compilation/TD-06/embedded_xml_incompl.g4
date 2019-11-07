/**
  * parsing an xml file
  */
grammar embedded_xml_incompl;

/* ############################################## */
/* #########      Syntaxical rules      ######### */
/* ############################################## */

file      : xml_prolog ? element* { System.out.println("XML file"); };
xml_prolog: '<?xml' attribute* '?>' { System.out.println("XML PROLOG"); };
attribute : QNAME '=' ATTRIB_VAL { System.out.println("ATTRIBUTE"); };
open      : '<' QNAME attribute* '>' { System.out.println("OPEN"); };
close     : '</' QNAME '>'{System.out.println("CLOSE"); } ;
openclose : '<' QNAME attribute* '/>' ;
element   : ( open ( element | content )* close | openclose ) { System.out.println("ELEMENT"); };
content   : ( NO_LESS | ATTRIB_VAL | QNAME )  { System.out.println("CONTENT"); };

/* ############################################## */
/* ##########       lexical rules      ########## */
/* ############################################## */

WS        : [ \t\r\n]+ -> skip ;
QNAME     : (ID ':')? ID ;
ATTRIB_VAL: '"' ~'"'* '"' | '\'' ~'\''* '\'' ;
NO_LESS   : ~'<' ;
fragment
ID      : [\p{Alphabetic}_] [-\p{Alphabetic}_0-9.]* ;
