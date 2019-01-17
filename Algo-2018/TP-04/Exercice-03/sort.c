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

void insertion_sort(int t[], int size)
{
  int i,j;

  for ( i = 0 ; i < size ; i++ )
  {
    for ( j=i-1 ; j >= 0 ; j-- )
    {
      if ( less(t[j+1], t[j]) )
      {
        swap(&t[j+1], &t[j]);
      }
    }
  }
}

int split_array(int t[], int size)
{
  int i,j;

  i = 1;
  j = size-1;

  while(1)
  {
    while( less(t[i], t[0]) && i < size-1)
    {
      i++;
    }
    while ( less(t[0], t[j]) && j > 0 )
    {
      j--;
    }
    if ( i >= j ) break;
    swap(&t[i], &t[j]);
    i++;
    j--;
  }
  swap(&t[0], &t[j]);

  return j;
}

void quick_sort(int t[], int size)
{
  int mid;

  if ( size < 2) return;

  mid = split_array(t, size);
  quick_sort(t,size-mid);
  quick_sort(&t[mid], size-mid+1);
}

