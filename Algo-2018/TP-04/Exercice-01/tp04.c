#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "array.h"
#include "sort.h"

#define MAX_VALUE 10000
#define MAX_SIZE 1000

int main(int argc, char *argv[]) {

	srand(time(NULL));

	int size = MAX_SIZE;
	int max_value = MAX_VALUE;

	/* tableau de référence */
	int* tab_ref = NULL;
	/* tableau de travail */
	int* tab = NULL;

	/* allocation et initialisation de la référence avec des valeurs aléatoires */
	tab_ref = create_array(size);
	fill_random_array(tab_ref, size, max_value);

	/* allocation et initialisation du tableau de travail avec les valeurs de référence */
	tab = create_array(size);
	copy_array(tab_ref, tab, size);

	print_array(tab, size);

	/* tri du tableau de travail */
	selection_sort(tab, size);
	
	print_array(tab, size);
	
	/* libération des tableaux */
	free(tab);
	tab = NULL;
	free(tab_ref);
	tab_ref = NULL;

	return EXIT_SUCCESS;
}
