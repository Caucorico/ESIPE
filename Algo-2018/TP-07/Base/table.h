#ifndef __TABLE_H__
#define __TABLE_H__

typedef struct _table {
	link **bucket;
	int M; /* nombre de seaux */
	int size; /* nombre de mots dans la table */
} table;

table* create_table( int M );

void add_occ_table(table *tab, char word[], int pos);

#endif