int pow_n(int p, int n)
{
  float test = 10.0f;
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

  printf("########################################\n");
  printf("TP-03 Exercice-01. \nBut : prendre comme parametre un entier a et un entier positif non nul n et retourne a^n. \n\n");

  for ( i = 0 ; i < 16 ; i++ )
  {
    printf("2^%d = %d \n", i, pow_n(2, i));
    printf("2^%d (rec) = %d \n", i, pow_n(2, i));
  }

  printf("\n\n########################################\n");
  return 0;
}
