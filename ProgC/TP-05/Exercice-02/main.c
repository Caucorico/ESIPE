#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
    printf("Vous allez creer un tableau. \nVeuillez rentrer la taille (positive) de ce tableau : ");
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

/* An empty main to test the compilation of the allocation and free
   functions. */
int main( void )
{

	int* test;
	int* test2;

  printf("########################################\n");
  printf("TP-05 Exercice-02. \nBut : Creer des fonctions de manipulation de tableau d'entiers finissant par -1. \n\n");

  /* Initialisation de l'aleatoire */
	srand(time(NULL));

  printf("Test de fill_array : \n");
  test = fill_array();
  printf("Tableau obtenu : \n");
  print_array(test);

  printf("\n\nTest de random_array sur un tableau de taille 5 pour des entiers inclut dans [[0, 30]] :\n");
	test2 = random_array(5, 30);
	print_array(test2);
  printf("\n\nMeme test sur un deuxieme tableau : \n");
	test = random_array(5, 30);
	print_array(test);

  printf("\n\nLa concatenation des deux tableaux precedants en utilisant concat_array donne : \n");
	test = concat_array(test, test2);
	print_array(test);

  return 0;
}
