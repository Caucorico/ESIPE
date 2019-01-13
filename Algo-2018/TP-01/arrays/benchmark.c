#include "benchmark.h"
#include "arrays.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * Perform a test with n_insert random insertions and n_find random calls
 * to find. Number are chosen from the interval [0..modulo] with a fixed
 * seed for the random number generator for replicability.
 */
int test_random(int nb, int mode, int n_insert, int n_find, int modulo);

int run_single_test(int nb, int mode) {

    switch (nb) {
        case 1  : return test_random(nb, mode, 10, 3, 20);
        case 2  : return test_random(nb, mode, 20, 10, 1);
        case 3  : return test_random(nb, mode, 0, 10, 10);
        case 4  : return test_random(nb, mode, 1, 10, 10);
        case 5  : return test_random(nb, mode, 10, 100, 20);
        case 6  : return test_random(nb, mode, 1000, 1000000, 2000);
        case 7  : return test_random(nb, mode, 10000, 20000, 20000);
        case 8  : return test_random(nb, mode, 30000, 60000, 60000);
        case 9  : return test_random(nb, mode, 60000, 120000, 120000);
        case 10 : return test_random(nb, mode, 100000, 200000, 50);
        default :
          return 0;
    }
}

int benchmark_unsorted(int nb) {
    return run_single_test(nb,0);
}

int benchmark_sorted(int nb) {
    return run_single_test(nb,MODE_SORTED);
}

int benchmark(int nb, int mode) {
    return run_single_test(nb,mode);
}

/*
 * Generate the correct answers for insertion
 */
void run_insert_ref(int *ins_arr, int ins_size, int *ins_corr,
                     void (*insert)(int*, int*, int)) {

    int size = 0;
    int i;

    printf("insertion (reference)\n");

	/* insertion */
    for (i = 0; i < ins_size; i++) {
        insert(ins_corr, &size, ins_arr[i]);
    }
}

/*
 * Generate the correct answers for find
 */
void run_find_ref(int *ins_corr, int ins_size, 
                     int *look_arr, int look_size, int *look_corr,
                     int (*find)(int*, int, int)) {

    int i;

    printf("find (reference)\n");

	/* find */
    for (i = 0; i < look_size; i++) {
        look_corr[i] = find(ins_corr, ins_size, look_arr[i]);
    }
}

/*
 * Test student functions
 */
int run_insert_student(int *ins_arr, int ins_size, int *ins_corr,
                       void (*insert)(int*, int*, int)) {

    int size = 0;
    int i, ok;
    int *arr = create_array(ins_size);

    printf("insertion (student)\n");

    /* insertion */
    for (i = 0; i < ins_size; i++) {
        insert(arr, &size, ins_arr[i]);
    }

    /* verify answers */
    ok = 1;
    for (i = 0; i < ins_size; i++) {
        if (arr[i] != ins_corr[i]) {
            printf("ERROR! position %d in array contains: %d, correct answer: %d\n", 
                i, arr[i], ins_corr[i]);
            ok = 0;
            break;
        }
    }
    if (ok) {
        printf("insertion ok\n");
    } else {
        printf("=====> insertion failed!\n");
    }

    free_array(arr);

    return ok;
}

/*
 * Test student functions
 */
int run_find_student(int *ins_corr, int ins_size, 
                        int *look_arr, int look_size, int *look_corr,
                        int (*find)(int*, int, int)) {

    int i, ok, res;

    printf("find (student)\n");

    ok = 1;
    for (i = 0; i < look_size; i++) {
        res = find(ins_corr, ins_size, look_arr[i]);
        if (res != look_corr[i]) {
            printf("ERROR! find (%d) returns: %d, correct answer: %d\n",
                look_arr[i], res, look_corr[i]);
            ok = 0;
            break;
        }
    }
    if (ok) {
        printf("find ok\n");
    } else {
        printf("=====> find failed!\n");
    }

    return ok;
}

/*
 * Display the number of milliseconds that has passed since start.
 */
void timer(clock_t start) {
    clock_t diff = clock()-start;
    int msec = diff*1000/CLOCKS_PER_SEC;
    printf("time: %d ms\n", msec);
}

int test_random(int nb, int mode, int n_insert, int n_find, int modulo) {

	int *ins_data = create_array(n_insert);
	int *look_data = create_array(n_find);
	int *ins_corr = create_array(n_insert);
	int *look_corr = create_array(n_find);

	int i;

    clock_t start;

    if (mode & MODE_SORTED)
        printf("*** TEST %d, SORTED ***\n", nb);
    else
        printf("*** TEST %d, UNSORTED ***\n", nb);
    printf("testing with random elements from 0..%d\n", modulo-1);
    printf("%d insertions, %d calls to find\n", n_insert, n_find);
    printf("*** BEGIN ***\n");

    /* fix seed */
	srand(42);

	/* generate insertion tests */
	for (i = 0; i < n_insert; i++) {
		ins_data[i] = rand() % modulo;
	}

	/* generate find tests */
	for (i = 0; i < n_find; i++) {
		look_data[i] = rand() % modulo;
	}

    void (*fn_insert_ref) (int*,int*,int);
    int  (*fn_find_ref)   (int*,int,int);
    void (*fn_insert)     (int*,int*,int);
    int  (*fn_find)       (int*,int,int);

    if (mode & MODE_SORTED) {
        fn_insert_ref = &insert_sorted_ref;
        fn_find_ref = &find_sorted_ref;
        fn_insert = &insert_sorted;
        fn_find = &find_sorted;
    } else {
        fn_insert_ref = &insert_unsorted_ref;
        fn_find_ref = &find_unsorted_ref;
        fn_insert = &insert_unsorted;
        fn_find = &find_unsorted;
    }
   
    int res1=1;
    int res2=1;

    start = clock();
	run_insert_ref(ins_data, n_insert, ins_corr, fn_insert_ref);
    timer(start);

    if (~mode & MODE_REFONLY) {
        start = clock();
        res1 = run_insert_student(ins_data, n_insert, ins_corr, fn_insert);
        timer(start);
    }

    start = clock();
    run_find_ref(ins_corr, n_insert, look_data, n_find, look_corr, fn_find_ref);
    timer(start);

    if (~mode & MODE_REFONLY) {
        start = clock();
        res2 = run_find_student(ins_corr, n_insert, look_data, n_find, look_corr, fn_find);
        timer(start);
    }

	printf("*** TEST %d END ***\n\n", nb);

	free_array(ins_data);
	free_array(look_data);
	free_array(ins_corr);
	free_array(look_corr);

	return res1 && res2;
}

