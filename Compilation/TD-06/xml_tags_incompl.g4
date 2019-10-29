/**
  * parsing xml tags
  */
grammar xml_tags;
file      : (NO_XML | NAME | VALUE | '/' | '=' | ':' | '>'
          | xml_tag { System.out.println(" tag parsed "); }
          | '</' | '<' )+ ;
xml_tag   : '<' name attribute* '/'? '>' | '</' name '>' ;
attribute : name '=' VALUE ;
name      : (NAME ':')? NAME ;
