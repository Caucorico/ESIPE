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
    printf("array = { ");
    for ( i = 0 ; i < array_size(array) ; i++ )
    {
      printf("%d, ", array[i]);
    }
    printf("}\n");
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

int* copy_array(int* array)
{
  int size, i;
  int* copy;

  size = array_size(array);
  copy = (int*)malloc(sizeof(int)*size);

  for ( i = 0 ; i < size ; i++ )
  {
    copy[i] = array[i];
  }
  copy[i+1] = -1;
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

int* fill_array(void)
{
  int size, i = 0, buf;
  int* array;

  do
  {
    printf("Vous allez creer un tableau. \n Veuillez rentrer la taille (positive) de ce tableau : ");
    scanf("%d", &size);
  }
  while( size < 0 );

  array = allocate_integer_array(size);

  while ( i < size )
  {
    printf("Veuillez rentrer l'entier %d sur %d : ", i+1, size);
    scanf("%d", &buf);
    if ( buf < 0 )
    {
      printf("Veuiller rentrer des nombres positifs.\n");
      continue;
    }
    array[i] = buf;
    i++;
  }
  array[i+1] = -1;

  return array;
}

/* An empty main to test the compilation of the allocation and free
   functions. */
int main(int argc, char* argv[]){

  return 0;
}
