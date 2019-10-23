#include "dirstat.h"

int get_dir_struct_stat(const char* path, struct dirent*** dir_stat)
{
	DIR* dir_stream;
	int err, nbr_element = 0;
	dir_stat* buff;

	dir_stream = opendir(path);
	if ( dir_stream == NULL )
	{
		perror("Error during the opendir ");
		return -1;
	}

	buff = readdir(dir_stream);
	while ( buff != NULL )
	{
		nbr_element++;
		buff = readdir(dir_stream);
	}

	rewinddir(dir_stream);
	*dir_stat = malloc(nbr_element*sizeof(struct dirent*));

	buff = readdir(dir_stream);
	while ( buff != null )
	{
		*dir_stat = buff;
		dir_stat++;
	}

	*dir_stat = readdir(dir_stream);
	if ( *dir_stat == NULL )
	{
		perror("Error during the readdir ");
		return -2;
	}

	err = closedir(dir_stream);
	if ( err != 0 )
	{
		perror("Error during the closedir ");
		return -3;
	}	

	return nbr_element;
}

int close_dir_struct_stat(const dirent*** dir_stat)
{
	if ( dir_stat == NULL ) return -1;
	free(*dir_stat);
	return 0;
}

int display_dir_name_struct_stat(const struct dirent* dir_stat[], int nbr_element)
{
	unsigned int i;

	for ( i = 0 ; i < nbr_element ; i++ )
	{
		printf("%s\n", dir_stat[i]->d_name);
	}

	return 1;
}

int display_dir_content_struct_stat(const struct dirent* dir_stat[], int nbr_element)
{
	int err;
	unsigned int i;
	struct stat file_stat buff;

	for ( i = 0 ; i < nbr_element ; i++ )
	{
		err = get_struct_stat(dir_stat[i]->d_name, &buff);
		if ( err < 0 )
		{
			fprintf(stderr, "Loop read file stat fail\n");
			continue;
		}
		display_struct_stat(&buff, dir_stat[i]->d_name);
	}

	return 1;
}

int display_dir_struct_stat(const struct dirent* dir_stat[], int nbr_element, unsigned char flags)
{
	if ( flags&DFN == DFN ) return display_dir_name_struct_stat(dir_stat, nbr_element);
	else if (flags&DCF == DCF ) return display_dir_content_struct_stat(dir_stat, nbr_element);
	else return 0;
}