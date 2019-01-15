#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

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

int* copy_cut_array(int* array, int size)
{
  int buf;
  int* tab;

  buf = array[size];
  array[size] = -1;
  tab = copy_array(array);
  array[size] = buf;
  return tab;
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
  array[i] = -1;

  return array;
}

int* random_array(int size, int max_entry)
{
	int i;
	int* tab;


	tab = allocate_integer_array(size);

	for ( i = 0 ; i < size ; i++ )
	{
		tab[i] = rand()%(max_entry+1);
	}

	tab[i] = -1;

	return tab;
}

int* concat_array(int* first, int* second)
{
	int* tab;
	int size1, size2;
	int i;

	size1 = array_size(first);
	size2 = array_size(second);

	tab = allocate_integer_array(size1+size2);

	for ( i = 0 ; i < size1 ; i++ )
	{
		tab[i] = first[i];
	}

	for ( i = 0 ; i < size2 ; i++ )
	{
		tab[i+size1] = second[i];
	}

	tab[i+size1] = -1;

	return tab;
}

int* merge_sorted_arrays(int* first, int* second)
{
  int i, j, k;
  int size1 , size2;
  int* sorted;

  size1 = array_size(first);
  size2 = array_size(second);

  sorted = allocate_integer_array(size1+size2);

  j = k = 0;
 
  for ( i = 0 ; j < size1 && k < size2; i++ )
  {
    if ( first[j] < second[k] )
    {
      sorted[i] = first[j];
      j++;
    }
    else
    {
      sorted[i] = second[k];
      k++;
    }
  }
  sorted[i] = -1;
  if ( j == size1 )
  {
    sorted = concat_array(sorted, &second[k]);
  }
  else
  {
    sorted = concat_array(sorted, &first[j]);
  }

  return sorted;
}

void split_arrays(int* array, int** first, int** second)
{
  int size;
  size = array_size(array);

  *first = copy_cut_array(array, (size/2)+(size%2));
  *second = copy_array(&array[(size/2)+(size%2)]); 
}

int* merge_sort(int* array)
{
  int* t1;
  int* t2;

  if ( array_size(array) == 1 )
  {
    return array;
  }

  split_arrays(array, &t1, &t2);

  return merge_sorted_arrays(merge_sort(t1),merge_sort(t2));
}

/* An empty main to test the compilation of the allocation and free
   functions. */
int main(int argc, char* argv[]){

	int* test;

	srand(time(NULL));

  test = random_array(100, 32000);
  print_array(test);

  test = merge_sort(test);
  print_array(test);

  free_integer_array(test);

  return 0;
}
