#include <stdio.h>

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

int pow_n_rec(int p, int n)
{
  if ( n == 0 ) return 1;
  return p*pow_n_rec(p, n-1);
}

int main(void)
{
  int i;
  for ( i = 0 ; i < 16 ; i++ )
  {
    printf("2^%d = %d \n", i, pow_n(2, i));
    printf("2^%d (rec) = %d \n", i, pow_n(2, i));
  }
  return 0;
}