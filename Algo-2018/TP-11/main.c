#include <stdio.h>
#include <time.h>
#include "heap.h"

void print_tab(int* tab, int size)
{
	int j;

	putchar('[');

	for ( j = 0 ; j < size ; j++ )
	{	
		if ( j == size-1 ) printf("%d", tab[j]);
		else printf("%d,", tab[j]);
	}

	printf("]\n");
}

void test_insert(heap* h)
{
	int i;

	srand(time(NULL));

	for ( i = 0 ; i < h->max ; i++ )
	{
		insert_heap(h, rand()%255);
	}

	print_tab(h->tree, h->size);
}

void test_extract_min(heap* h)
{
	int i, min;

	srand(time(NULL));

	for ( i = 0 ; i < h->max ; i++ )
	{
		insert_heap(h, rand()%255);
	}

	print_tab(h->tree, h->size);

	for ( i = 0 ; i < h->max ; i++ )
	{
		min = extract_min(h);
		printf("min = %d\n", min);
		print_tab(h->tree, h->size);
	}
}

int main( void )
{
	heap* h;
	int i, min;
	int tab[10] = { 54, 654, 8, 55, 23, 666, 77, 856, 32, 8 };

	/*h = create_heap(10);

	test_insert(h);

	free_heap(h);*/

	heapsort(tab, 10);
	print_tab(tab, 10);

	return 0;
}