#ifndef _TOP10F_
#define _TOP10F_

typedef struct _files
{
  long int size;
  char* path;
  struct _files* next;
}files;

typedef struct _t10f
{
  files* first;
  long int size;
  long int floor;
}t10f;

t10f* get_top10f(const char* path);

void free_files( files* f );

void free_top10f( t10f* top10f );

#endif