#include <stdio.h>
#include <stdlib.h>

int is_numerical( char *string )
{
  int i=0;

  if( string[i] == '\0' ) return 0;

  while ( string[i] != '\0' )
  {
    if ( string[i] < '0' || string[i] > '9' )
    {
      return 0;
    }
    i++;
  }

  return 1;
}

int is_characters( char *string )
{
  int i=0;

  if( string[i] == '\0' ) return 0;

  while ( string[i] != '\0' )
  {
    if ( string[i] < 'a' || string[i] > 'z' )
    {
      return 0;
    }
    i++;
  }

  return 1;
}

void invert_order( char *string )
{
  int i=0, j=0;
  char buff;

  while ( string[i] != '\0' ) i++;
  j = i-1;
  i=0;
  while ( i < j)
  {
    buff = string[i];
    string[i] = string[j];
    string[j] = buff;
    i++;
    j--;
  }

}

void convert_into_base_26 ( char *string )
{
  int nbr, i=0;

  nbr = atoi(string);

  while ( nbr/26 != 0 )
  {
    string[i] = (nbr%26) + 'a';
    i++;
    nbr /= 26;
  }

  string[i] = (nbr%26) + 'a';
  string[i+1] = '\0';

  invert_order(string);

}

int pow_n(int p, int n)
{
  int i=1, res=p;
  if ( n == 0 ) return 1;
  while( i < n )
  {
    res *= p;
    i++;
  }
  return res;
}

int convert_into_base_10 ( char *string )
{
  int nbr=0, i=0, size;

  while ( string[i] != '\0' )i++;
  size = i-1;

  i=0;
  while ( string[i] != '\0' )
  {
    nbr += (string[i]-'a')*pow_n(26 , size);
    size--;
    i++;
  }

  return nbr;
}

int main( int argc, char** argv )
{
  int test,i;

  printf("########################################\n");
  printf("TP-03 Exercice-03. \nBut : Convertir les nombre de base 10 en base 26 et vice versa. \n\n");

  /* Zone TP */

  for ( i = 1 ; i < argc ; i++ )
  {
    if ( is_characters(argv[i]) )
    {
      test = convert_into_base_10(argv[i]);
      printf("%s base 10 = %d \n", argv[i], test);
    }
    else if ( is_numerical( argv[i] ) )
    {
      printf("%s base 26 = ", argv[i]);
      convert_into_base_26(argv[i]);
      printf("%s\n", argv[i]);
    }
    else
    {
      printf("erreur d'argument\n");
    }
  }

  /* Fin zone TP */

  printf("\n\n########################################\n");
  return 0;
}