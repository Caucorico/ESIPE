#include <stdlib.h>
#include <stdio.h>
#include "heap.h"

heap* create_heap(int max)
{
	heap* new_heap = malloc(sizeof(heap));
	if ( new_heap == NULL )
	{
		perror("Malloc error in create_heap in heap.c ");
		return NULL;
	}

	new_heap->size = 0;
	new_heap->max = max;

	new_heap->tree = malloc(sizeof(int)*max);
	if ( new_heap->tree == NULL )
	{
		perror("Malloc error in create_heap in heap.c ");
		free(new_heap);
		return NULL;
	}

	return new_heap;
}

void free_heap(heap* h)
{
	if ( h != NULL )
	{
		if ( h->tree != NULL )
		{
			free(h->tree);
		}
		free(h);
	}
}

void insert_heap(heap* h, int elt)
{
	int i, j, buff;

	if ( h == NULL )
	{
		fprintf(stderr, "Error in insert_heap in heap.c : h is NULL ! \n");
		return;
	}
	else if ( h->tree == NULL )
	{
		fprintf(stderr, "Error in insert_heap in heap.c : h->tree is NULL ! \n");
		return;
	}
	else if ( h->size >= h->max )
	{
		fprintf(stderr, "Error in insert_heap in heap.c : h is full !\n");
		return;
	}

	h->tree[h->size] = elt;
	h->size++;

	j = h->size-1;
	i = (j-1)/2;
	
	while ( j > 0 && h->tree[i] > h->tree[j] )
	{
		buff = h->tree[j];
		h->tree[j] = h->tree[i];
		h->tree[i] = buff;
		j = i;
		i = (j-1)/2;
	}
}

int is_heap(heap* h)
{
	int i;

	if ( h == NULL )
	{
		fprintf(stderr, "Error in is_heap in heap.c : h is NULL ! \n");
		return 0;
	}
	else if ( h->tree == NULL )
	{
		fprintf(stderr, "Error in is_heap in heap.c : h->tree is NULL ! \n");
		return 0;
	}
	
	for ( i = 0 ; i < h->size ; i++ )
	{
		if ( i*2 + 1 >= h->size ) return 1;
		else if ( h->tree[i*2 + 1] < h->tree[i] ) return 0;	
		else if ( i*2 + 2 < h->size && h->tree[i*2 + 2] < h->tree[i] ) return 0;
	}

	return 1;
}

int min_child(heap* h, int index)
{
	if ( h == NULL )
	{
		fprintf(stderr, "Error in min_child in heap.c : h is NULL ! \n");
		return -1;
	}
	else if ( h->tree == NULL )
	{
		fprintf(stderr, "Error in min_child in heap.c : h->tree is NULL ! \n");
		return -1;
	}

	if ( index*2 + 1 >= h->size ) return -1;
	else if ( index*2 + 2 >= h->size ) return index*2 + 1;
	else if ( h->tree[index*2 + 2] < h->tree[index*2 + 1] ) return index*2 + 2;
	else return index*2 + 1;
}

int extract_min(heap* h)
{
	int i, j, buff, min;

	if ( h == NULL )
	{
		fprintf(stderr, "Error in extract_min in heap.c : h is NULL ! \n");
		return -1;
	}
	else if ( h->tree == NULL )
	{
		fprintf(stderr, "Error in extract_min in heap.c : h->tree is NULL ! \n");
		return -1;
	}

	min = h->tree[0];
	h->tree[0] = h->tree[h->size-1];
	h->size--;

	i = 0;

	while ( min_child(h, i) != -1 && h->tree[min_child(h, i)] < h->tree[i] )
	{
		j = min_child(h, i);
		buff = h->tree[i];
		h->tree[i] = h->tree[j];
		h->tree[j] = buff;
		i = j;
	}

	return min;
}

void heapsort(int tab[], int size)
{
	int i, buff;
	heap h;

	h.tree = tab;
	h.size = 0;
	h.max = size;

	for ( i = 0 ; i < size ; i++ )
	{
		insert_heap(&h, tab[i]);
	}

	for ( i = 0 ; i < size ; i++ )
	{
		buff = extract_min(&h);
		tab[size-i] = buff;
	}
}