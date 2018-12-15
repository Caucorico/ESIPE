#include <stdio.h>
#include <stdlib.h>

int main( int argc, char *argv[] )
{
  int a, b;

  if ( argc != 3 )
  {
    printf("usage : %s <entier 1> <entier 2>\n", argv[0]);
    return 1;
  }

  a = atoi( argv[1] );
  b = atoi( argv[2] );
  printf("La somme des deux entiers est : %d. \n", a+b );
  return 0;
}