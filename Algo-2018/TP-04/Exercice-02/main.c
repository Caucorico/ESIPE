#include <stdlib.h>
#include <stdio.h>

int main( void )
{
  int values[] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
  int t1[], t2[];
  int i;

  for (i=0; i < 10; i++)
    printf("%d ", values[i]);
  printf("\n");
  
  qsort (values, 10, sizeof(int), compare);
  
  for (i=0; i < 10; i++)
    printf ("%d ", values[i]);
  printf("\n");

  return 0;
}
}