/** A Javadoc-style comment on the grammar
 */
 grammar td1_ex1;

/** A rule that matches the entire file
 *  A Javadoc-style comment on the rule
 */
log  : (TIME | WS | EOL | OTHER)* ;

/** A TIME is three fields with two digits each */
TIME : [0-9][0-9] ':' [0-9][0-9] ':' [0-9][0-9] ;
OTHER : ~[0-9 \t\r\n]+ | ~[: \t\r\n]+ ;  // Define graphic stuff
WS  :   [ \t\r]+ ; // Define whitespace rule, toss it out
EOL  :   [\n] ;
