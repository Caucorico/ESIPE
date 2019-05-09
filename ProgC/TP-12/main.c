#include <stdio.h>
#include "word_list.h"
#include "parser.h"
#define NB_PACK 4096

int main(int argc, char** argv)
{
	node* list[NB_PACK];

	if ( argc != 2 )
	{
		fprintf(stderr, "usage : %s <file_name>\n", argv[0] );
		return -1;
	}

	get_file_word_list(argv[1]);

	if ( list != NULL )
	{
		printf("%d different words found in Germinal\n", get_list_size(list));
		return 0;
	}

	return -2;

}