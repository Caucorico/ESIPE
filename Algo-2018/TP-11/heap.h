#ifndef _HEAP_
#define _HEAP_

typedef struct _heap
{
	int* tree;
	int size;
	int max;
}heap;

heap* create_heap(int max);

void free_heap(heap* h);

void insert_heap(heap *h, int elt);

int is_heap(heap* h);

int min_child(heap* h, int index);

int extract_min(heap* h);

void heapsort(int tab[], int size);

#endif