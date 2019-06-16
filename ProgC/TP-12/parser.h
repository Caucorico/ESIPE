#ifndef _PARSER_
#define _PARSER_

#include <stdio.h>
#include "word_list.h"

void build_word_list(FILE* f, node** list);

void build_word_hash_tab(FILE* f, node** list);

node* get_file_word_list(char* name);

void get_file_word_hash_tab(char* name, node** hash_tab);

#endif