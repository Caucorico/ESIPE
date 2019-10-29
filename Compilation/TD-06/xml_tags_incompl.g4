/**
  * parsing xml tags
  */
grammar xml_tags_incompl;
file      : (NO_XML | NAME | VALUE | '/' | '=' | ':' | '>'
          | xml_tag { System.out.println(" tag parsed "); }
          | '</' | '<' )+ ;
xml_tag   : '<' name attribute* '/'? '>' | '</' name '>' ;
attribute : name '=' VALUE ;
name      : (NAME ':')? NAME ;


NAME : [a-zA-Z]+[a-zA-Z0-9\-]*[/]? ;
VALUE : '"' NAME  '"' ;
NO_XML : ~[<] -> skip ;
WHITE_SPACE : [ \n\r\t] -> skip ;
