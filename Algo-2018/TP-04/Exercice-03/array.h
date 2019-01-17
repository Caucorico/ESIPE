#ifndef ARRAY_H
#define ARRAY_H

int* create_array(int size);

void print_array(int t[], int size);

void copy_array(int src[], int dst[], int size);

void fill_random_array(int t[], int size, int max_value);

/**
 * Initialises the array t with a random permutation of
 *   the values 0, 1, ..., size-1.
 * Uses the Knuth shuffle:
 *   https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
 */
void fill_random_permutation(int t[], int size);

int compare_array(int t1[], int t2[], int size);

#endif /* ARRAY_H */
