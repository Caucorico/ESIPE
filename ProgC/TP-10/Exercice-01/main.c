#include <stdlib.h>
#include <stdio.h>

void swap_mem(void* z1, void* z2, size_t size)
{
  char* cz1 = (char*)z1;
  char* cz2 = (char*)z2;
  unsigned char buff;
  size_t i;

  for( i = 0 ; i < size ; i++ )
  {
    buff = cz1[i];
    cz1[i] = cz2[i];
    cz2[i] = buff;
  }
}

int main(void)
{
  char t1[10] = { 'c', 'o', 'u', 'c', 'o', 'u', '\0' };
  char t2[10] = { 'w', 'o', 'r', 'l', 'd', '\0' };

  printf("%s %s \n", t1, t2);
  swap_mem(t1, t2, 10);
  printf("%s %s \n", t1, t2);

  return 0;
}
