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
	int res=0, i,j;

	/* tableau de référence */
	int* tab_ref = NULL;

	/* tableau de qsort */
	int* tab_qsort = NULL;

	/* tableau de travail. */
	int* tab;

	tab_ref = create_array(size);
	tab_qsort = create_array(size);
	tab = create_array(size);

	printf("test tri insertion : \n");

	/* test avec beaucoup de taille de tableau différente. */
	for ( i = 0 ; i < size ; i++ )
	{
		/* initialisation de la référence avec des valeurs aléatoires */
		fill_random_array(tab_ref, i, max_value);

		/* initialisation du tableau de travail avec les valeurs de référence */
		copy_array(tab_ref, tab, i);

		/* initialisation du tableau qsort avec les valeurs de référence */
		copy_array(tab_ref, tab_qsort, i);

		/* tri du tableau de travail */
		insertion_sort(tab, i);
		qsort(tab_qsort, i, sizeof(int), compare);

		res += compare_array(tab, tab_qsort, i);
	}

	printf("Notre fonction de tri a obtenue %d resultat identiques a qsort  sur %d pour des tableaux aléatoire de taille %d a %d.\n", res, size, 0, size-1);

	res = 0;
	for ( i = 0; i < size ; i++)
	{
		for ( j = 0 ; j < i ; j++ )
		{
			tab_ref[j] = j;
		}

		copy_array(tab_ref, tab_qsort, i);

		copy_array(tab_ref, tab, i);

		insertion_sort(tab, i);
		qsort(tab_qsort, i, sizeof(int), compare);

		res += compare_array(tab, tab_qsort, i);

	}

	printf("Notre fonction de tri a obtenue %d resultat identiques a qsort  sur %d pour des tableaux tries de taille %d a %d.\n", res, size, 0, size-1);

	res = 0;
	for ( i = 0; i < size ; i++)
	{
		for ( j = i ; j >= 0 ; j-- )
		{
			tab_ref[j] = j;
		}

		copy_array(tab_ref, tab_qsort, i);

		copy_array(tab_ref, tab, i);

		insertion_sort(tab, i);
		qsort(tab_qsort, i, sizeof(int), compare);

		res += compare_array(tab, tab_qsort, i);

	}

	printf("Notre fonction de tri a obtenue %d resultat identiques a qsort  sur %d pour des tableaux inverses de taille %d a %d.\n", res, size, 0, size-1);

	res = 0;
	for ( i = 0; i < size ; i++)
	{
		for ( j = i ; j >= 0 ; j-- )
		{
				tab_ref[j] = j%2;
		}

		copy_array(tab_ref, tab_qsort, i);

		copy_array(tab_ref, tab, i);

		insertion_sort(tab, i);
		qsort(tab_qsort, i, sizeof(int), compare);

		res += compare_array(tab, tab_qsort, i);

	}

	printf("Notre fonction de tri a obtenue %d resultat identiques a qsort  sur %d pour des tableaux repetitifs de taille %d a %d.\n", res, size, 0, size-1);

	printf("test tri rapide : \n");

	/* test avec beaucoup de taille de tableau différente. */
	for ( i = 0 ; i < size ; i++ )
	{
		/* initialisation de la référence avec des valeurs aléatoires */
		fill_random_array(tab_ref, i, max_value);

		/* initialisation du tableau de travail avec les valeurs de référence */
		copy_array(tab_ref, tab, i);

		/* initialisation du tableau qsort avec les valeurs de référence */
		copy_array(tab_ref, tab_qsort, i);

		/* tri du tableau de travail */
		quick_sort(tab, i);
		qsort(tab_qsort, i, sizeof(int), compare);

		res += compare_array(tab, tab_qsort, i);
	}

	printf("Notre fonction de tri a obtenue %d resultat identiques a qsort  sur %d pour des tableaux aléatoire de taille %d a %d.\n", res, size, 0, size-1);

	res = 0;
	for ( i = 0; i < size ; i++)
	{
		for ( j = 0 ; j < i ; j++ )
		{
			tab_ref[j] = j;
		}

		copy_array(tab_ref, tab_qsort, i);

		copy_array(tab_ref, tab, i);

		quick_sort(tab, i);
		qsort(tab_qsort, i, sizeof(int), compare);

		res += compare_array(tab, tab_qsort, i);

	}

	printf("Notre fonction de tri a obtenue %d resultat identiques a qsort  sur %d pour des tableaux tries de taille %d a %d.\n", res, size, 0, size-1);

	res = 0;
	for ( i = 0; i < size ; i++)
	{
		for ( j = i ; j >= 0 ; j-- )
		{
			tab_ref[j] = j;
		}

		copy_array(tab_ref, tab_qsort, i);

		copy_array(tab_ref, tab, i);

		quick_sort(tab, i);
		qsort(tab_qsort, i, sizeof(int), compare);

		res += compare_array(tab, tab_qsort, i);

	}

	printf("Notre fonction de tri a obtenue %d resultat identiques a qsort  sur %d pour des tableaux inverses de taille %d a %d.\n", res, size, 0, size-1);

	res = 0;
	for ( i = 0; i < size ; i++)
	{
		for ( j = i ; j >= 0 ; j-- )
		{
				tab_ref[j] = j%2;
		}

		copy_array(tab_ref, tab_qsort, i);

		copy_array(tab_ref, tab, i);

		quick_sort(tab, i);
		qsort(tab_qsort, i, sizeof(int), compare);

		res += compare_array(tab, tab_qsort, i);

	}

	printf("Notre fonction de tri a obtenue %d resultat identiques a qsort  sur %d pour des tableaux repetitifs de taille %d a %d.\n", res, size, 0, size-1);

	/* libération des tableaux */
	free(tab);
	tab = NULL;
	free(tab_ref);
	tab_ref = NULL;
	free(tab_qsort);
	tab_qsort = NULL;

	return EXIT_SUCCESS;
}
