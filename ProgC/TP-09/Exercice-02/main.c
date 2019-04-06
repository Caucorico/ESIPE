#include <stdio.h>
#include <stdlib.h>

char* allocate_char_array(int size)
{
	return (char*)malloc(size*sizeof(char));
}

char** allocate_char_poiter_array(int size)
{
	return (char**)malloc(size*sizeof(char*));
}

char** allocate_and_create_two_dim_array( int size1, int size2 )
{
	int i;
	char** array;
	array = allocate_char_poiter_array(size1);
	if ( array == NULL ) return NULL;

	for ( i = 0 ; i < size1 ; i++ )
	{
		array[i] = allocate_char_array(size2);
	}

	return array;
}

void free_char_array( char* array )
{
	free(array);
}

void free_char_pointer_array( char** array, int size )
{
	int i;

	for ( i = 0 ; i < size ; i++ )
	{
		free_char_array(array[i]);
	}
	free(array);
}

void fill_and_display_char_array( char** array, int size1, int size2 )
{
	int i,j;

	for ( i = 0 ; i < size1 ; i++ )
	{
		for ( j = 0 ; j < size2 ; j++ )
		{
			array[i][j] = ('a'+(i+j)%26);
			printf("%c ", array[i][j]);
		}
		putchar('\n');
	}
}

int main(int argc, char** argv)
{
	int size1, size2;
	char** array;

	if ( argc != 3 )
	{
		printf("Wrong parameters, usage : %s <tab_size_dim_1> <tab_size_dim_2>\n", argv[0]);
		return -1;
	}

	size1 = atoi(argv[1]);
	size2 = atoi(argv[2]);

	array = allocate_and_create_two_dim_array(size1, size2);
	if ( array == NULL )
	{
		perror("The tab was not allocated !");
		return -2;
	}
	fill_and_display_char_array(array, size1, size2);
	free_char_pointer_array(array, size1);

	return 0;
}