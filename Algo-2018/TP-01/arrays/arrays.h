#ifndef ARRAYS_H
#define ARRAYS_H

/*
 * Create an array of size max_size.
 * Return a pointer to the array if succesful, otherwise 0.
 */
int *create_array(int max_size);

/*
 * Free the memory of an array.
 */
void free_array(int t[]);

/*
 * Insert the element elt in an unsorted array t.
 * The value of *size is updated.
 */
void insert_unsorted(int t[], int *size, int elt);

/*
 * Find the element elt in an unsorted array t using linear search.
 * Return 1 if the element is present, 0 otherwise.
 */
int find_unsorted(int t[], int size, int elt);

/*
 * Insert the element elt in a sorted array t.
 * The value of *size is updated.
 */
void insert_sorted(int t[], int *size, int elt);

/*
 * Find the element elt in a sorted array t using binary search.
 * Return 1 if the element is present, 0 otherwise.
 */
int find_sorted(int t[], int size, int elt);

#endif /* ARRAYS_H */