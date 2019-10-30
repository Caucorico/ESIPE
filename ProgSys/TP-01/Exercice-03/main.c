#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "stat.h"
#include "dirstat.h"


int main( int argc, char** argv )
{
	int err;
	struct dirent** dir_stat;

	if ( argc != 2 )
	{
		fprintf(stderr, "Usage : %s <dir_path>\n", argv[0]);
		return 1;
	}

	err = get_dir_struct_stat(argv[1], &dir_stat);
	if ( err < 0 )
	{
		perror("Error during get_dir_struct_stat ");
		return 2;
	}

	err = display_dir_struct_stat(dir_stat, err, 0x1);
	if ( err < 0 )
	{
		perror("Error during display_dir_struct_stat ");
		err = close_dir_struct_stat(&dir_stat);
		if ( err == -1 ) fprintf(stderr, "The dir_stat is NULL !\n");
		return 3;
	}
	
	err = close_dir_struct_stat(&dir_stat);
	if ( err == -1 ) fprintf(stderr, "The dir_stat is NULL ! \n");

	return 0;
}
