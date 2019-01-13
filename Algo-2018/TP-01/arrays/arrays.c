#include "arrays.h"
#include <stdlib.h>

int *create_array(int max_size) {
	int *ptr = (int*)malloc(max_size*sizeof(int));
	return ptr;
}

void free_array(int t[]) {
	free(t);
}

/*
 * Write this function!
 */
void insert_unsorted(int t[], int *size, int elt) {
	t[*size] = elt;	
	(*size)++;
}

/*
 * Write this function!
 */
int find_unsorted(int t[], int size, int elt) {
	int i;
	for ( i = 0 ; i < size ; i++ )
	{
		if ( t[i] == elt )
		{
			return 1;
		}
	}
	return 0;
}

/*
 * Write this function!
 */
void insert_sorted(int t[], int *size, int elt) {
	int i = *size;
	(*size)++;
	while ( t[i-1] > elt && i-1 >= 0)
	{
		t[i] = t[i-1];
		i--;
	}
	t[i] = elt;
}

/*
 * Write this function!
 */
int find_sorted(int t[], int size, int elt) {
	int mid = size/2;
	if ( size == 0 )
  {
		return 0;
	}
	else if( t[mid] == elt )
  {
		return 1;
	}
	else if ( t[mid] > elt )
  {
		return find_sorted(t, mid, elt);
	}
	else /* ( t[mid]<elt ) */
  {
		return find_sorted( t+(mid+1), size-(mid+1), elt);
	}
}