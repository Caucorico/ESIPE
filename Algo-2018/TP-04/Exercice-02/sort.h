#ifndef SORT_H
#define SORT_H

int less(int a, int b);

void swap(int *a, int *b);

void selection_sort(int t[], int size);

void insertion_sort(int t[], int size);

void quick_sort(int t[], int size);

int compare(const void *a,const void *b);

#endif /* SORT_H */