#include <stdio.h>
#include <stdlib.h>

/* Fonction qui affiche la liste decroissant-croissante de maniere interative. */
void display_liste( int k )
{
  int i;
  for ( i = k ; i > 0 ; i-- )
  {
    printf("%d ", i);
  }
  for ( i = 1 ; i <= k ; i++ )
  {
    printf("%d ", i);
  }
}

/* Fonction qui affiche la liste decroissante-croissante de maniere recursive. */
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

  printf("########################################\n");
  printf("TP-01 Exercice-03. \nBut : Afficher la suite  n n-1 n-2 .. 0 .. n-2 n-1 n de facon iterative puis recursif. \n\n");

  /* Zone TP */
  if ( argc != 2 )
  {
    printf("Usage : %s <entier> \n", argv[0]);
    return 1;
  }

  k = atoi( argv[1] );

  printf("Resultat fonction iterative :\n");
  display_liste(k);
  printf("\nResultat fonction recursive  :\n");
  display_liste_recursif(k);

  /* Fin zone TP */

  printf("\n\n########################################\n");

  return 0;
}