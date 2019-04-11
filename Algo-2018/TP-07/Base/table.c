#include <stdlib.h>
#include <stdio.h>
#include "table.h"
#include "list.h"

table* create_table( int M )
{
	int i;
	table* new_table = malloc(sizeof(table));
	if ( new_table == NULL ) return NULL;

	new_table->M = M;
	new_table->size = 0;
	new_table->bucket = malloc(sizeof(link*)*M);

	for ( i = 0 ; i < M ; i++ )
	{
		new_table->bucket[i] = NULL;
	}

	return new_table;
}

unsigned int hash(char* elt)
{
	int i;
	unsigned int h;
	h = 0;

	for ( i = 0 ; elt[i] != '\0' ; i++ )
	{
		h = 31*h + elt[i];
	}

	return h;
}

void add_occ_table(table *tab, char word[], int pos)
{
	int bucket;
	link* occ;

	bucket = hash(word)%tab->M;
	occ = find_list(tab->bucket[bucket], word);

	if ( occ == NULL )
	{
		tab->bucket[bucket] = insert_first_list(tab->bucket[bucket], word, pos);
	}
	else
	{
		add_occurrence( occ, pos);
	}
}

void free_table(table* tab)
{
	int i;

	for ( i = 0 ; i < tab->M ; i++ )
	{
		if ( tab->bucket[i] != NULL )
		{
			free_list( tab->bucket[i] );
		}
	}

	free(tab);
}

void display_table(table* tab)
{
	int i;

	for ( i = 0 ; i < tab-> M ; i++ )
	{
		if ( tab->bucket[i] != NULL )
		{
			printf("bucket[%d] => ", i);
			display_list(tab->bucket[i]);
		}
	}
}

int size_table(table *tab)
{
	int i;
	int total_number_words = 0;

	for ( i = 0 ; i < tab->M ; i++ )
	{
		if ( tab->bucket[i] != NULL )
		{
			total_number_words += get_total_list_word_number(tab->bucket[i]);
		}
	}

	return total_number_words;
}