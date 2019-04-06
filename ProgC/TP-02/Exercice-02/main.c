#include <stdio.h>
#include <stdlib.h>

int main( int argc, char *argv[] )
{
  int a, b;

  printf("########################################\n");
  printf("TP-02 Exercice-02. \nBut : Afficher la somme de deux entiers passés en parametre. \n\n");

  /* Zone TP */
  if ( argc != 3 )
  {
    printf("usage : %s <entier 1> <entier 2>\n", argv[0]);
    return 1;
  }

  a = atoi( argv[1] );
  b = atoi( argv[2] );
  printf("La somme des deux entiers est : %d. \n", a+b );
  /* Fin zone TP */

  printf("########################################\n");

  return 0;
}