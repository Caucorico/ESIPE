#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <strings.h>

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
  char* fn = NULL;
  char* ln = NULL;

  if ( new_cell == NULL ) return NULL;

  if ( first != NULL ) 
  {
  	fn = malloc(strlen(first)+1);
  	fn = strcpy(fn, first);
  }

  if ( last != NULL ) 
  {
  	ln = malloc(strlen(last)+1);
  	ln = strcpy(ln, last);
  }

  new_cell->first_name = fn;
  new_cell->last_name = ln;
  new_cell->age = age;
  new_cell->next = NULL;

  return new_cell;
}

/* Insert p2 into p1 if possible.
 * Insert in age order
 * Return 1 if the insertion was done or 0 if not
 * P1 & P2 not null
 */
/*int age_order(Cell* p1, Cell* p2)
{
  if ( p1 == NULL ) return -1; /* What are you trying ?!? 
  if ( p2 == NULL ) return -1;
  if ( (p1->age < p2->age) && ( (p1->next == NULL) || (p1->next->age > p2->age) ) )
  {
    p2->next = p1->next;
    p1->next = p2;
    return 1;
  }    
  else if ( p2->age < p1->age )
  {
    swap_mem(p1, p2, sizeof(Cell));
    p1->next = p2;
    return 1;
  }
    
  return 0;
}*/

int age_order(Cell* p1, Cell* p2)
{
  if ( p1 == NULL ) return -1; /* What are you trying ?!? */
  if ( p2 == NULL ) return -1;

  if ( p1->age < p2->age )
    return -1;

  if ( p1->age == p2->age )
  	return 0;

  if ( p1->age > p2->age )
  	return 1;
}

/* Insert p2 into p1 if possible.
 * Insert in name order
 * Return 1 if the insertion was done or 0 if not
 * P1 & P2 not null
 */
int name_order(Cell* p1, Cell* p2)
{
	char* full_name_p1;
	char* full_name_p2;
	int ret;

	if ( p1 == NULL ) return -1; /* What are you trying ?!? */
  if ( p2 == NULL ) return -1;

	full_name_p1 = calloc( strlen(p1->first_name) + strlen(p1->last_name) ,sizeof(char));
	full_name_p2 = calloc( strlen(p2->first_name) + strlen(p2->last_name) ,sizeof(char));
  full_name_p1 = strcpy(full_name_p1, p1->first_name);
  full_name_p2 = strcpy(full_name_p2, p2->first_name);
  full_name_p1 = strcat(full_name_p1, p1->last_name);
  full_name_p2 = strcat(full_name_p2, p2->last_name);

  if ( strcmp(full_name_p1, full_name_p2) < 0 )
	  ret = -1;

  if ( strcmp(full_name_p1, full_name_p2) == 0 )
  	ret = age_order(p1,p2);

  if ( strcmp(full_name_p1, full_name_p2) > 0 )
  	ret = 1;

  free(full_name_p1);
  free(full_name_p2);

  return ret;
}

void ordered_insertion(List* l, Cell* new, int order_func(Cell*, Cell*))
{
  Cell* buff = *l;

  if ( buff == NULL )
  {
    *l = new;
    return;
  }
  
  /*while( order_func(buff, new) < 1 ) buff = buff->next;*/

  while ( buff != NULL )
  {
  	if ( ( order_func(buff, new) <= 0  ) && ( (buff->next == NULL) || (order_func(buff->next, new) >= 0) ) )
		{
		  new->next = buff->next;
		  buff->next = new;
		  return;
		}    
		else if ( order_func(buff, new) == 1  )
		{
		  swap_mem(buff, new, sizeof(Cell));
		  buff->next = new;
		  return;
		}
		else
		{
			buff = buff->next;
		}
  }
}

void print_list(List l)
{
  while( l != NULL )
  {
    printf("%s %s %d \n", l->first_name, l->last_name, l->age);
    l = l->next;
  }
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
	FILE* f;
  List l = NULL;
  Cell* b;
  char fn[100], ln[100];
  int age;

  f = fopen("liste_nom.txt", "r");
  if ( f == NULL )
  {
  	perror("Error open file ");
  	return -1;
  }

  bzero(fn, 100*sizeof(char));
  bzero(ln, 100*sizeof(char));
  
  srand(time(NULL));

  while ( fscanf(f, "%s %s %d", fn, ln, &age ) != EOF )
  {
    b = allocate_cell(fn, ln, age );
    ordered_insertion(&l, b, name_order );

    bzero(fn, 100*sizeof(char));
  	bzero(ln, 100*sizeof(char));
  }

  print_list(l);

  free_list(l);

  fclose(f);
  
  return 0;
}
