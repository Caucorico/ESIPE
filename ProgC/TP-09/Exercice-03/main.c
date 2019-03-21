#include <stdio.h>
#include <stdlib.h>

char* copy_string_array( char *array  )
{
	int size=0, i;
	char* cp_array;
	while ( array[size] != '\0' ) size++;
	cp_array = (char*) malloc(size*sizeof(char));
	for ( i = 0 ; i < size+1 ; i++ )
	{
		cp_array[i] = array[i]; 
	}
	return cp_array;
}

char** do_the_same_job_as_argv(int argc, char** argv)
{
	int i;
	char** array;

	array = (char**) malloc(argc*sizeof(char*));
	if ( array == NULL ) return NULL;

	for ( i = 0 ; i < argc ; i++ )
	{
		array[i] = copy_string_array(argv[i]);
	}

	return array;
}

void free_string_array(char *array)
{
	free(array);
}

void free_argv_like(char **array, int size)
{
	int i;
	for ( i = 0 ; i < size ; i++ )
	{
		free_string_array(array[i]);
	}
	free(array);
}

void display_argv(int size, char** array)
{
	int i;

	for ( i = 0 ; i < size ; i++ )
	{
		printf("argv[%d] = '%s'\n", i, array[i]);
	}
}

int main(int argc, char** argv)
{
	char** argv_like;

	argv_like = do_the_same_job_as_argv(argc, argv);
	if ( argv_like == NULL )
	{
		perror("The tab was not allocated !");
		return -1;
	}

	display_argv(argc, argv_like );

	free_argv_like(argv_like, argc);
	return 0;
}