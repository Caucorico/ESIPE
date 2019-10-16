#include "stat.h"

int get_struct_stat(const char* path, struct stat* file_stat)
{
	int err;

	err = stat(path, file_stat);

	if ( err == -1 )
	{
		perror("Error in the stat execution ");
		return err;
	}

	return 0;

}

int display_struct_stat(const struct stat* file_stat)
{
	if ( file_stat == NULL )
	{
		fprintf(stderr, "display_struct_stat cannot take NULL file_stat\n");
		return -1;
	}

	printf("Inode number : %ld\n", file_stat->st_ino);
	printf("Size : %ld\n", file_stat->st_size);
	printf("Last modification : %ld\n", file_stat->st_mtime);

	if ( S_ISDIR(file_stat->st_mode) )
	{
		printf("This file is a dir\n");
	}
	else if ( S_ISREG(file_stat->st_mode) )
	{
		printf("This file is a regular file\n");
	}
	else
	{
		printf("Other type\n");
	}

	return 0;
}


