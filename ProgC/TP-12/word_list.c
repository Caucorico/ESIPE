#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "word_list.h"

char* create_word(char* w)
{
	char* new_word = (char*)malloc(sizeof(char)*(strlen(w)+1));
	if ( new_word == NULL )
	{
		perror("Error malloc in create_word() in word_list.c");
		return NULL;
	}

	strcpy(new_word, w);

	return new_word;
}

node* create_word_node(char* w)
{
	node* new_cell;

	if ( w == NULL ) return NULL;

	new_cell = (node*)malloc(sizeof(node));
	if ( new_cell == NULL )
	{
		perror("Error malloc in create_word_cell() in word_list.c");
		return NULL;
	}

	new_cell->next = NULL;
	new_cell->word = create_word(w);

	if ( new_cell->word == NULL )
	{
		fprintf(stderr, "The word in the node structure in create_word_node() in word_list.c is NULL ! \n The cell is destroy to keep integrity.\n");
		free(new_cell);
		return NULL;
	}

	return new_cell;
}

void free_word(char* w)
{
	if ( w != NULL )
	{
		free(w);	
	}
}

node* get_word_in_list(node* list, char* w)
{
	if ( w == NULL )
	{
		fprintf(stderr, "The word in get_word_in_list in word_list.c is NULL ! \n");
		return NULL;
	}

	while ( list != NULL && strcmp(w, list->word) != 0 )
		list = list->next;

	return list;
}

int insert_word_in_first(node** list, char* w)
{
	node* new_cell;

	if ( w == NULL )
	{
		fprintf(stderr, "The word in insert_word_in_first in word_list.c is NULL ! \n");
		return -1;
	}


	if ( get_word_in_list(*list, w) != NULL )
	{
		return 0;
	}

	new_cell = create_word_node(w);

	if ( list == NULL )
	{
		*list = new_cell;
		return 1;
	}

	new_cell->next = *list;
	*list = new_cell;

	return 2;
}

int get_list_size(node* list)
{
	unsigned int size = 0;

	while ( list != NULL )
	{
		size++;
		list = list->next;
	}

	return size;
}