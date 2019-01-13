#include <stdio.h>
#include <stdlib.h>

void display_liste( int k )
{
  int i;
  for ( i = k ; i > 0 ; i-- )
  {
    printf("%d ", i);
  }
  for ( i = 1 ; i < k ; i++ )
  {
    printf("%d ", i);
  }
}

void display_liste_recursif( int k )
{
  if ( k == 0 ) return;
  printf("%d ", k);
  display_liste_recursif( k-1 );
  printf("%d ", k);
}

int main( int argc, char *argv[] )
{
  int k;

  if ( argc != 2 )
  {
    printf("Usage : %s <entier> \n", argv[0]);
  }

  k = atoi( argv[1] );
  /* display_liste(k); */
  display_liste_recursif(k);

  return 0;
}