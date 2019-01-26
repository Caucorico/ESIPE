#include <stdio.h>
#include <stdlib.h>

int array_size(int* array)
{
  int i=0;
  if ( array == NULL ) return 0;
  while( array[i] != -1 ) i++;
  return i;
}

void print_array(int* array)
{
  int i = 0;
  if ( array == NULL )
  {
    printf("array = NULL\n");
  }
  else
  {
    printf("array = {");
    for ( i = 0 ; i < array_size(array) ; i++ )
    {
      printf(" %d", array[i]);
      if ( i < array_size(array)-1)
      {
        putchar(',');
      }
    }
    printf(" }\n");
  }
}

int are_arrays_equal(int* first, int* second)
{
  int i;

  if( first == second )
  {
    return 1;
  }
  else if ( first == NULL || second == NULL )
  {
    return 0;
  }
  else if ( array_size(first) != array_size(second) )
  {
    return 0;
  }
  else
  {
    for ( i = 0 ; i < array_size(first) ; i++ )
    {
      if ( first[i] != second[i] )
      {
        return 0;
      }
    }
    return 1;
  }
}

/* Allocate memory for an array which can contain `size`
   integers. The returned C array has memory for an extra last
   integer labelling the end of the array. */
int* allocate_integer_array(int size){
  int* new_tab;

  new_tab = (int*)malloc((size+1)*sizeof(int));
  if (new_tab == NULL){
    fprintf(stderr, "Memory allocation error\n");
    return NULL;
  }
  return new_tab;
}

/* Free an integer array */
void free_integer_array(int* tab){
  free(tab);
}

int* copy_array(int* array)
{
  int size, i;
  int* copy;

  size = array_size(array);
  copy = allocate_integer_array(size);

  for ( i = 0 ; i < size ; i++ )
  {
    copy[i] = array[i];
  }
  copy[i] = -1;

  return copy;
}

/* An empty main to test the compilation of the allocation and free
   functions. */
int main(void)
{
  int array[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1 };
  int array2[4] = { 2, 3, 4, -1 };
  int array3[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1 };
  int* array4;
  printf("########################################\n");
  printf("TP-05 Exercice-01. \nBut : Creer des fonctions de manipulation de tableau d'entiers finissant par -1. \n\n");

  /* Zone TP */

  printf("Test de print array pour le tableau qui represente la suite [[0, 10]] :\n");
  print_array(array);
  printf("\nLa taille de ce tableau d'apres array_size est : %d\n\n", array_size(array));
  printf("Comparaison de { 2, 3, 4 } avec [[0, 10]]. \nLa fonction are_array_equal retourne : %d\n\n", are_arrays_equal(array, array2));
  printf("Comparaison de [[0, 10]] avec [[0, 10]]. \nLa fonction are_array_equal retourne : %d\n\n", are_arrays_equal(array, array3));
  array4 = allocate_integer_array(11);
  array4 = copy_array(array);
  printf("La copie de array 1 vers array4 donne : \n");
  print_array(array4);

  /* Fin zone TP */

  printf("\n\n########################################\n");
  return 0;
}
