#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void init_random( void )
{
  srand(time(NULL));
}

void gen_random_array( int* array, int size, int val_min, int val_max)
{
  int i;

  for ( i = 0 ; i < size ; i++ )
  {
    array[i] = val_max - (rand()%(val_min+1));
  }
}

void display_array(int* array, int size)
{
  int i;
  printf("[");

  for (i = 0 ; i < size ; i++ )
  {
    printf(" %d", array[i]);
    if ( i < size-1) putchar(',');
  }

  printf(" ]\n");
}

void bubble_sort( int* array, int size)
{
  int state=1, i, buff;
  while ( state )
  {
    state=0;
    for ( i = 1 ; i < size ; i++ )
    {
      if ( array[i-1] > array[i] )
      {
        buff = array[i];
        array[i] = array[i-1];
        array[i-1] = buff;
        state = 1;
      }
    }
  }
}

int is_sort( int* array, int size )
{
  int i;
  for ( i = 0 ; i < size-1 ; i++ )
  {
    if ( array[i] > array[i+1] )
    {
      return 0;
    }
  }
  return 1;
}

int main( void )
{
  int test[10], i;

  printf("########################################\n");
  printf("TP-03 Exercice-05. \nBut : Trier un tableau avec un tri a bulles. \n\n");

  /* Zone TP */

  init_random();
  for ( i = 0 ; i < 10 ; i++ )
  {
    gen_random_array( test, i, 5, 10);
    display_array(test, i);
    bubble_sort(test, i);
    printf("res : \n");
    display_array(test, i);
    putchar('\n');
    printf("--------------------------------------\n\n");
  }

  /* Fin zone TP */

  printf("\n\n########################################\n");
  return 0;
}