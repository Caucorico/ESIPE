grammar exercice02;

@parser::members {
	float info = 0;
	float error = 0;
	float notice = 0;
	int lines = 0;
}

log          : line* {
	float other = lines - (info + error + notice);
	System.out.println("[notice] " + (notice/lines)*100 + "%" );
	System.out.println("[error] " + (error/lines)*100 + "%" );
	System.out.println("[info] " + (info/lines)*100 + "%" );
	System.out.println("[other] " + (other/lines)*100 + "%" );
};

line         : { lines++; } date_block info_block? client_block?  STRING+? ENDLINE ;
date_block   : '[' TEXT TEXT NUMBER TIME NUMBER ']' ;
info_block   : '[' TEXT ']' { switch($TEXT.text){
	case "notice" :
		notice++;
		break;
	case "info" :
		info++;
		break;
	case "error" :
		error++;
		break;			
}; } ;
client_block : '[' TEXT IPV4 ']' ;


NUMBER: [0-9]+ ;
TEXT  : [\p{Alphabetic}]+ ;
TIME  : [0-9][0-9] ':' [0-9][0-9] ':' [0-9][0-9] {/* System.out.println("time : " + getText());*/ };
WS    : [ \t\r]+ -> skip ;
ENDLINE : [\n\r]+ ;
IPV4  : [0-9]+'.'[0-9]+'.'[0-9]+'.'[0-9]+ ;
STRING : ~[\n] ;
/* EOF   : EOF -> skip ; */
