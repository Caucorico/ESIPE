/**
  * parsing an xml file
  */
grammar embedded_xml;
file      : xml_prolog ? element ;
xml_prolog: '<?xml' attribute * '?>' ;
element   : 
(...)
WS        : [ \t\r\n]+ -> skip ;
QNAME     : (ID ':')? ID ;
ATTRIB_VAL: '"' ~'"'* '"' | '\'' ~'\''* '\'' ;
NO_LESS   : ~'<' ;
fragment
ID      : [\p{Alphabetic}_] [-\p{Alphabetic}_0-9.]* ;