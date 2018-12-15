#include <stdio.h>

int main(void)
{
  FILE* file_stream;
  char a;

  file_stream = fopen("main.c","r");
  if( file_stream == NULL )
  {
    perror("open file error : ");
    return -1;
  }

  while( fread(&a, 1, 1, file_stream) > 0 )
  {
    putchar(a);
  }
  fclose(file_stream);

  return 0;
}