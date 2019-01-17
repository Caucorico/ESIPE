#include <stdio.h>
#include <stdlib.h>
#include "sort.h"

int less(int a, int b) {
	return a < b;
}

void swap(int *a, int *b) {
	int tmp = *a;
	*a = *b;
	*b = tmp;
}

void selection_sort(int t[], int size)
{
	int i, j;
  int min;
  for ( i = 0 ; i < size ; i++ )
  {
    min = i;
    for (j = i+1 ; j < size ; j++)
    {
      if ( less(t[j], t[min]))
        min = j;
    }
    swap(&t[i], &t[min]);
  }
}

int compare(const void *a,const void *b)
{
  int aint = *(int*)a;
  int bint = *(int*)b;
  if (aint == bint)
    return 0;
  else
    if (aint < bint)
      return -1;
    else
      return 1;
}

