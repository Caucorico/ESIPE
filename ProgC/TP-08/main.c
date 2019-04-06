#include <stdio.h>
#include <stdlib.h>

long int cache[200000000];

void syracuse( unsigned long int val )
{
  while ( val != 1 )
  {
    printf("%ld ", val);
    if ( val%2 )
    {
      val = (val*3) + 1;
    }
    else
    {
      val = val/2;
    }
  }
  printf("%ld ", val);
}

unsigned long int fly_length_syracuse( unsigned long int val )
{
  unsigned long int i=0;

  while ( val != 1 )
  {
    if ( val%2 )
    {
      val = (val*3) + 1;
    }
    else
    {
      val = val/2;
    }
    i++;
  }

  return i;
}

unsigned long int max_fly_length_syracuse( unsigned long int val_max )
{
  unsigned long int i, max=0, buf;

  for ( i = 1 ; i <= val_max ; i++ )
  {
    buf = fly_length_syracuse(i);
    if ( buf > max ) max = buf;
  }

  return max;

}


unsigned long int rec_fly_length_syracuse( unsigned long int val )
{
  if ( val == 1 ) return 0;
  if ( cache[val] != -1  ) return cache[val];
  if ( val%2 )
    {
      cache[val] = 1+rec_fly_length_syracuse((val*3)+1);
      return cache[val];
    }
    else
    {
      cache[val] = 1+rec_fly_length_syracuse(val/2);
      return cache[val];
    }
}

unsigned long int rec_max_fly_length_syracuse( unsigned long int val_max )
{
  unsigned long int i, max=0, buf;

  for ( i = 1 ; i <= val_max ; i++ )
  {
    buf = rec_fly_length_syracuse(i);
    if ( buf > max ) max = buf;
  }

  return max;

}

void init_cache( void )
{
  unsigned long int i;
  for ( i = 0 ; i < 200000000 ; i++ )
  {
    cache[i] = -1;
  }
}


int main( int argc, char** argv )
{
  unsigned long int t;
  if ( argc != 2 )
  {
    printf("Usage : %s <n>\n", argv[0]);
    return 1;
  }
  init_cache();

  t = rec_max_fly_length_syracuse(atoi(argv[1]));
  printf("\n Max fly Length : %lu\n", t);
  t = max_fly_length_syracuse(atoi(argv[1]));
  printf("\n Max fly Length : %lu\n", t);
  return 0;
}