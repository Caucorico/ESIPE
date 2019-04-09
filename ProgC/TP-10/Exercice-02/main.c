#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct cell
{
  char* first_name;
  char* last_name;
  int age;
  struct cell* next;
}Cell, *List;

void swap_mem(void* z1, void* z2, size_t size )
{
  char* cz1 = (char*)z1;
  char* cz2 = (char*)z2;
  unsigned char buff;
  size_t i;

  for ( i = 0 ; i < size ; i++ )
  {
    buff = cz1[i];
    cz1[i] = cz2[i];
    cz2[i] = buff;
  }
}

Cell* allocate_cell(char* first, char* last, int age)
{
  Cell* new_cell = (Cell*)malloc(sizeof(Cell));
  if ( new_cell == NULL ) return NULL;
  new_cell->first_name = first;
  new_cell->last_name = last;
  new_cell->age = age;
  new_cell->next = NULL;

  return new_cell;
}

/* Insert p2 into p1 if possible.
 * Return 1 if the insertion was done or 0 if not
 * P2 not null
 */
int age_order(Cell* p1, Cell* p2)
{
  if ( p1 == NULL ) return -1; /* What are you trying ?!? */
  if ( p2 == NULL ) return -1;
  if ( (p1->age < p2->age) && ( (p1->next->age > p2->age) || (p1->next == NULL) ) )
  {
    p2->next = p1->next;
    p1->next = p2;
    return 1;
  }    
  else if ( p2->age < p1->age )
  {
    swap_mem(p1, p2, sizeof(Cell)); /* segfault here */
    p1->next = p2;
    return 1;
  }
    
  return 0;
}

int name_order(Cell* p1, Cell* p2)
{
  return 0;
}

void ordered_insertion(List* l, Cell* new, int order_func(Cell*, Cell*))
{
  Cell* buff = *l;

  if ( buff == NULL )
  {
    *l = new;
    return;
  }
  
  while( !order_func(buff, new) ) buff = buff->next;
}

void print_list(List l)
{
  while( l != NULL )
  {
    printf("%d,", l->age);
    l = l->next;
  }
  putchar('\n');
}

void free_list(List l)
{
  Cell* buff;

  while ( l != NULL )
  {
    buff = l;
    l = l->next;
    
    if ( buff->first_name != NULL )
      free(buff->first_name);
    
    if ( buff->last_name != NULL )
      free(buff->last_name);
    
    free(buff);
  }
}

int main(void)
{
  List l;
  Cell* b;
  int i;
  
  srand(time(NULL));

  for ( i = 0 ; i < 32 ; i++ )
  {
    b = allocate_cell(NULL, NULL, rand()%256 );
    ordered_insertion(&l, b, age_order );
  }

  print_list(l);

  free_list(l);
  
  return 0;
}
