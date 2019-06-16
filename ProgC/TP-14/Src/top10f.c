#include <stdlib.h>
#include <stdio.h>
#include <ftw.h>
#include <string.h>
#include "top10f.h"

static t10f* top10f;

t10f* create_top10f( void )
{
  t10f* new_top;

  new_top = malloc( sizeof(t10f) );
  if ( new_top == NULL )
  {
    perror("Error in top10f.c in create_top10f ");
    return NULL;
  }

  new_top->first = NULL;
  new_top->size = 0;
  new_top->floor = 0;

  return new_top;

}

void free_files( files* f )
{
  if ( f != NULL )
  {
    if ( f->path != NULL ) free(f->path);
    free(f);
  }
}

void free_top10f( t10f* top10f )
{
  files* buff;
  files* buff2;

  if ( top10f != NULL )
  {
    buff = top10f->first;
    while ( buff != NULL )
    {
      buff2 = buff;
      buff = buff->next;
      free_files(buff2);
    }
    free(top10f);
  }
  
}

int insert_top10f( t10f* list, const struct stat* sb, const char* path )
{
  files* f;
  files* buff;
  files* buff2;

  if ( list == NULL )
  {
    fprintf(stderr, "Error in top10f.c in insert_top10f : list is NULL\n");
    return -1;
  }
  else if ( sb == NULL )
  {
    fprintf(stderr, "Error in top10f.c in insert_top10f : sb is NULL\n" );
    return -2;
  }
  else if ( path == NULL )
  {
    fprintf(stderr, "Error in top10f.c in insert_top10f : path is NULL\n");
    return -3;
  }

  f = malloc( sizeof(files) );
  if ( f == NULL )
  {
    perror("Error in top10f.c in insert_top10f ");
    return -1;
  }
  f->size = sb->st_size;
  f->path = malloc( (strlen(path)*sizeof(char) ) + 1 );
  strcpy(f->path, path);

  if ( list->size == 0 )
  {
    list->first = f;
    list->size++;
    list->floor = sb->st_size;
  }
  else if ( list->size < 10 )
  {
    if ( list->first->size > f->size )
    {
      f->next = list->first;
      list->first = f;
      list->size++;
      list->floor = f->size;
    }
    else
    {
      buff = list->first;
      while ( buff->next != NULL && buff->next->size < sb->st_size ) buff = buff->next;
      f->next = buff->next;
      buff->next = f;
      list->size++;
    }
  }
  else
  {
    if ( sb->st_size > list->floor )
    {
      buff = list->first;
      while ( buff->next != NULL && buff->next->size < sb->st_size ) buff = buff->next;
      f->next = buff->next;
      buff->next = f;
      buff2 = list->first;
      list->first = list->first->next;
      list->floor = list->first->size;
      free_files(buff2);
    }
    else
    {
      free_files(f);
    }
  }

  return 0;
}

int navigate(const char* fpath, const struct stat *sb, int typeflag)
{
  if ( fpath == NULL )
  {
    fprintf(stderr, "Error in top10f.c in navigate(): the path cannot be NULL.\n");
    return -1;
  }
  else if ( sb == NULL )
  {
    fprintf(stderr, "Error in top10f.c in navigate(): the sb cannot be NULL.\n");
    return -2;
  }
  else if ( typeflag < 0 )
  {
    fprintf(stderr, "Error in top10f.c in get_top10f(): the sb cannot be NULL.\n");
    return -3;
  }

  if ( typeflag == FTW_F )insert_top10f(top10f, sb, fpath);

  return 0;
}

t10f* get_top10f(const char* path)
{
  if ( path == NULL )
  {
    fprintf(stderr, "Error in top10f.c in get_top10f : path is NULL\n");
    return NULL;
  }

  top10f = create_top10f();

  ftw(path,navigate,-1);

  return top10f;
}