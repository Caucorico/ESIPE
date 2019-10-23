#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "stat.h"


int main( int argc, char** argv )
{
	int err;
	struct stat file_stat;

	if ( argc != 2 )
	{
		fprintf(stderr, "Usage : %s <file_path>\n", argv[0]);
		return 1;
	}

	err = get_struct_stat(argv[1], &file_stat);

	if ( err < 0 )
	{
		fprintf(stderr, "An error is occur, program stops :(\n");
		return 2;
	}

	display_struct_stat(&file_stat, argv[1]);

	return 0;
}
