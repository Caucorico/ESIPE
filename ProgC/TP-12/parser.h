#ifndef _PARSER_
#define _PARSER_

#include <stdio.h>
#include "word_list.h"

void build_word_list(FILE* f, node** list);

node* get_file_word_list(char* name);

#endif