#ifndef BENCHMARK_H
#define BENCHMARK_H

#define N_TESTS 10

#define MODE_SORTED  0x0001
#define MODE_REFONLY 0x0002

int benchmark_unsorted(int nb);
int benchmark_sorted(int nb);
int benchmark(int nb, int mode);

/*
 * Reference functions.
 */
void insert_unsorted_ref(int t[], int *size, int elt);
int find_unsorted_ref(int t[], int size, int elt);
void insert_sorted_ref(int t[], int *size, int elt);
int find_sorted_ref(int t[], int size, int elt);

#endif /* BENCHMARK_H */