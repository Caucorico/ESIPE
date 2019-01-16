#include <stdio.h>
#include <stdlib.h>
#include "sort.h"
#include "array.h"
#include "visualarray.h"

#define MAX_SIZE 320
#define WIDTH 640

void bubble_sort_simple(int tab[], int size) {
	int i, k;
	for (i = 0; i < size; i++) {
		for (k = size-1; k > i; k--) {
			if (tab[k-1] > tab[k]) {
				swap(&tab[k-1], &tab[k]);
			}
		}

		/* Draw the array after one full iteration */
		visualize();
		MLV_wait_milliseconds(10);
	}
}

int main() {

	int *tab = create_array(MAX_SIZE);
	fill_random_permutation(tab, MAX_SIZE);

	/* open a window and link tab to it */
	init_visual(tab, MAX_SIZE, WIDTH, 0);

	bubble_sort_simple(tab, MAX_SIZE);
	MLV_wait_milliseconds(2000);

	/* close the window */
	free_visual();

	free(tab);

	return EXIT_SUCCESS;
}