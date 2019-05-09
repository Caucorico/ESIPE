#include "parser.h"

void build_word_list(FILE* f, node** list)
{
	char w[100];


	while ( fscanf(f, "%s", w) != EOF )
	{
		insert_word_in_first(list, w);
	}

}

void build_word_hash_tab(FILE* f, node** list)
{
	char w[100];
	int h;

	while ( fscanf(f, "%s", w) != EOF )
	{
		h = hash(w);
		insert_word_in_first(list[h], w);
	}

}

node* get_file_word_list(char* name)
{
	FILE* f = fopen(name, "r");
	node* list = NULL;

	if ( f == NULL )
	{
		perror("fopen file failed in get_file_word_list() in parser.c");
		return NULL;
	}

	build_word_list(f, &list);

	fclose(f);

	return list;
}

void get_file_word_hash_tab(char* name, node** hash_tab)
{
	FILE* f = fopen(name, "r");
	node** list = NULL;

	if ( f == NULL )
	{
		perror("fopen file failed in get_file_word_hash_tab() in parser.c");
		return NULL;
	}

	build_word_hash_tab(f, hash_tab);

	fclose(f);
}