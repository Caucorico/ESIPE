#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "array.h"
#include "sort.h"

#define MAX_VALUE 10000
#define MAX_SIZE 1000

extern int nb_less;
extern int nb_swap;

int main(int argc, char *argv[])
{

  FILE* file;
  unsigned int i;
  int* tab1 = NULL;
  int* tab2 = NULL;
  int* tab3 = NULL;
  int buf_nb_less, buf_nb_swap;

  file = fopen("sort.dat", "w");

  nb_less = 0;
  nb_swap = 0;

  for ( i = 0 ; i <= 10000 ; i+=100 )
  {
    nb_less = 0;
    nb_swap = 0;

    tab1 = create_array(i);
    tab2 = create_array(i);
    tab3 = create_array(i);

    fill_random_array(tab1,i,50);

    copy_array(tab1, tab2, i);
    copy_array(tab1, tab3, i);

    selection_sort(tab2, i);

    buf_nb_less = nb_less;
    buf_nb_swap = nb_swap;

    nb_less = 0;
    nb_swap = 0;

    insertion_sort(tab3, i);

    fprintf(file, "%d %d %d\n", i, buf_nb_swap, nb_swap  );

    free(tab1);
    free(tab2);
    free(tab3);

  }



	printf("%d comparaisons\n", nb_less);
  printf("%d échanges\n", nb_swap);

  fclose(file);

	return EXIT_SUCCESS;
}
