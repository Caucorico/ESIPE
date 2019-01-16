#include <stdio.h>
#include <stdlib.h>
#include "sort.h"
#include "array.h"
#include "visualarray.h"

#define MAX_SIZE 15
#define WIDTH 640

void bubble_sort_simple(int tab[], int size) {
	int i, k;
	for (i = 0; i < size; i++) {
		for (k = size-1; k > i; k--) {

			/**
			 * Draw the pair of values tab[k-1], tab[k]
			 *   in GREEN if they are in the correct order
			 *   in RED if they are not in the correct order
			 */
			if (tab[k-1] > tab[k])
				visualize_2_positions(k-1, k, MLV_COLOR_RED);
			else
				visualize_2_positions(k-1, k, MLV_COLOR_GREEN);
			MLV_wait_milliseconds(500);
			/* */

			if (tab[k-1] > tab[k]) {
				swap(&tab[k-1], &tab[k]);

				/**
				 * The pair tab[k-1] and tab[k] are now in the correct order,
				 *   so we draw them in GREEN.
				 */
				visualize_2_positions(k-1, k, MLV_COLOR_GREEN);
				MLV_wait_milliseconds(500);
				/* */
			}
		}
	}

	/* Draw the sorted array in the foreground colour */
	visualize();
}

int main() {

	int *tab = create_array(MAX_SIZE);
	fill_random_permutation(tab, MAX_SIZE);

	/* open a window and link tab to it */
	init_visual(tab, MAX_SIZE, WIDTH, 1);

	bubble_sort_simple(tab, MAX_SIZE);
	MLV_wait_milliseconds(2000);

	/* close the window */
	free_visual();

	free(tab);

	return EXIT_SUCCESS;
}