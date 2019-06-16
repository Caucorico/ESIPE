#include <stdio.h>
#include "top10f.h"

int main( int argc, char** argv )
{
  t10f* result;
  files* buff;

  if ( argc != 2 )
  {
    fprintf(stderr, "The command needs only one argument : the path\n");
    return 1;
  }

  result = get_top10f( argv[1] );

  if ( result == NULL )
  {
    fprintf(stdout, "The command doesn't found any result\n");
    return 2;
  }

  buff = result->first;

  while ( buff != NULL )
  {
    fprintf(stdout, "%s, size : %ld\n",buff->path, buff->size );
    buff = buff->next;
  }

  free_top10f( result );

  return 0;
}