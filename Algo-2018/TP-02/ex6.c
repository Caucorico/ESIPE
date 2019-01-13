#include <stddef.h>

int longest_incr_iter(int t[], int lo, int hi)
{
  int i;
  int size_iter_max = 0, current_size = 0;

  if ( t == NULL )
  {
    return 0;
  }


  for ( i = lo ; i <= hi-1 ; i++ )
  {
    if ( t[i] < t[i+1] )
    {
      current_size++;
    }
    else
    {
      if ( current_size >= size_iter_max )
      {
        current_size++;
        size_iter_max = current_size;
      }
      current_size=0;
    }
  }
  if ( current_size >= size_iter_max )
  {
    current_size++;
    size_iter_max = current_size;
  }

  return size_iter_max;
}

int first_incr(int t[], int lo, int hi)
{
  if ( lo > hi )
  {
    return 0;
  }
  if ( t[lo] < t[lo+1] )
  {
    return 1+first_incr(t, lo+1, hi);
  }
  else
  {
    return 1;
  }
}

int longest_incr_rec(int t[], int lo, int hi)
{
  int x, y;
  if ( lo > hi )
  {
    return 0;
  }
  else
  {
    x = first_incr(t, lo, hi);
    y = longest_incr_rec(t, lo+x , hi);
    if ( x < y )
    {
      return y;
    }
    else
    {
      return x;
    }
  }
}