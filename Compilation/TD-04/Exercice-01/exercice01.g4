lexer grammar exercice01;

CROCHET : '['.*?']'
	{ System.out.println("<"+getText()+">"); } ;

OTHER : .->skip ;
