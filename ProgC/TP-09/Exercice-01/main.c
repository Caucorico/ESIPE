#include <stdio.h>
#include <stdlib.h>

void* allocate_int_tab( int size )
{
	return (int*)malloc(size*sizeof(int));
}

void free_tab( int* tab )
{
	free(tab);
}

void init_and_display_incr_array( int* tab, int size )
{
	int i;

	for ( i = 0 ; i < size ; i++ )
	{
		tab[i] = i;
		printf("%d ", tab[i]);
	}
	putchar('\n');
}


int main(int argc, char** argv)
{
	int size;
	int* tab;

	if ( argc != 2 )
	{
		printf("Wrong parameters, usage : %s <tab_size>\n", argv[0]);
		return -1;
	}

	size = atoi(argv[1]);

	tab = allocate_int_tab(size);
	if ( tab == NULL )
	{
		perror("The tab was not allocated !");
		return -2;
	}

	init_and_display_incr_array(tab, size);

	free_tab(tab);
	return 0;
}