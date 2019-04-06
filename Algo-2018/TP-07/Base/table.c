#include "table.h"

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
		h = 31*h = elt[i];
	}

	return h;
}

void add_occ_table(table *tab, char word[], int pos)
{

}