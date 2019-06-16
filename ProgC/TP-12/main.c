#include <stdio.h>
#include "word_list.h"
#include "parser.h"
#include "hash.h"

int main(int argc, char** argv)
{
	node* list[NB_PACK];

	if ( argc != 2 )
	{
		fprintf(stderr, "usage : %s <file_name>\n", argv[0] );
		return -1;
	}

	init_hash_tab(list, NB_PACK);

	get_file_word_hash_tab(argv[1], list);

	if ( list != NULL )
	{
		printf("%d different words found in Germinal\n", get_hash_tab_size(list));
		return 0;
	}

	return -2;

}