#include <stdio.h>
#include <stdlib.h>
#include "array.h"

int* create_array(int size) {
	int* t = (int*) malloc(size * sizeof(int));	
	if(t == NULL) {
		printf("Problème allocation mémoire\n");
		exit(EXIT_FAILURE);
	}
	return t;
}

void print_array(int t[], int size) {
	int i;
	printf("[");
	for(i=0;i<size-1;i++) {
		printf("%d, ",t[i]);	
	}
	if(size>0)
		printf("%d",t[size-1]);
	printf("]\n");
	
}

void copy_array(int src[], int dst[], int size) {
  int i;
  for (i=0;i<size;i++) {
    dst[i]=src[i];
  }
}

void fill_random_array(int t[], int size, int max_value) {
	int i;
	for(i=0; i<size; i++) {
		t[i] = rand()%max_value;
	}
}

void fill_random_permutation(int t[], int size) {
    int i;
    for (i = 0; i < size; i++) {
        int j = rand()%(i+1);
        t[i] = t[j];
        t[j] = i;
    }
}

int compare_array(int t1[], int t2[], int size)
{
	int i;
	for ( i = 0 ; i < size ; i++ )
	{
		if ( t1[i] != t2[i] )
		{
			return 0;
		}
	}
	return 1;
}



