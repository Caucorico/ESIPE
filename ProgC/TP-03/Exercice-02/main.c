#include <stdio.h>

struct file_data
{
  int w;
  int c;
  int l;
};

unsigned char is_espace(char c)
{
  if ( c == '\n' || c == '\t' || c == ' ' || c == '\0' )
  {
    return 1;
  }
  else
  {
    return 0;
  }
}

unsigned char is_line(char c)
{
  if ( c == '\n' || c == '\0' ) return 1;
  else return 0;
}

/* word count in a file */
struct file_data wc(FILE* file_stream)
{
  char a,b;
  struct file_data file_data;
  file_data.c = 0;
  file_data.w = 0;
  file_data.l = 0;

  if ( fread( &a, 1, 1, file_stream ) < 1 )
  {
    return file_data;
  }
  while ( 1 )
  {
    b = a;
    file_data.c++;
    if ( fread( &a, 1, 1, file_stream ) < 1 )
    {
      file_data.l++;
      if ( !is_espace(b) ) file_data.w++;
      break;
    }
    if ( is_line(b) )
    {
      file_data.l++;
    }
    if ( !is_espace(b) && is_espace(a) )
    {
      file_data.w++;
    }

  }

  return file_data;
}

struct file_data wc_stdin( void )
{
  char a,b;
  struct file_data file_data;
  file_data.c = 0;
  file_data.w = 0;
  file_data.l = 0;

  a = fgetc(stdin);

  while( a > 0 )
  {
    b = a;
    file_data.c++;
    a = fgetc(stdin);
    fseek(stdin,0,SEEK_END);
    if ( !is_espace(b) && is_espace(a) )
    {
      file_data.w++;
    }
    if ( is_line(a) )
    {
      file_data.l++;
    }
  }

  return file_data;
}

int main( void )
{
  struct file_data file_data;

  file_data = wc_stdin();
  printf("c : %d, w: %d, l: %d \n", file_data.c, file_data.w, file_data.l );
  return 0;
}