#include <stdio.h>


/* Renvoie l'indice de value si present dans le tableau sinon renvoie -1. */
int binary_search( int* array, int size, int value )
{
  int buff;
  if ( array[size/2] == value ) return size/2;
  else if ( size == 1  ) return -1;
  else
  {
    if ( value < array[size/2] )
    {
      return binary_search(array, size/2, value);
    }
    else
    {
      buff = binary_search(&array[size/2], (size/2)+(size%2), value);
      if ( buff == -1 )
      {
        return -1;
      }
      return buff+(size/2);
    }
  }
}

int main(void)
{
  int test1[10] = { 0 , 5 , 6 , 7, 8 , 9 };
  printf("########################################\n");
  printf("TP-03 Exercice-04. \nBut : Effectuer une recherche dichotomique dans un tableau trie. \n\n");

  /* Zone TP */

  printf("Rechercher 6 dans { 0 , 5 , 6 , 7, 8 , 9 }, indice : %d\n", binary_search(test1, 6, 6));
  printf("Rechercher 9 dans { 0 , 5 , 6 , 7, 8 , 9 }, indice : %d\n", binary_search(test1, 6, 9));
  printf("Rechercher -5 dans { 0 , 5 , 6 , 7, 8 , 9 }, indice : %d\n", binary_search(test1, 6, -5));
  printf("Rechercher 3 dans { 0 , 5 , 6 , 7, 8 , 9 }, indice : %d\n", binary_search(test1, 6, 3));
  printf("Rechercher 0 dans { 0 , 5 , 6 , 7, 8 , 9 }, indice : %d\n", binary_search(test1, 6, 0));
  printf("Rechercher 20 dans { 0 , 5 , 6 , 7, 8 , 9 }, indice : %d\n", binary_search(test1, 6, 20));

  /* Fin zone TP */

  printf("\n\n########################################\n");
  return 0;
}