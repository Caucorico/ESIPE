#include <stdlib.h>
#include "body_list.h"
#include "body.h"

body_list* create_body_list( void )
{
	body_list* new_body_list = malloc(sizeof(body_list));
	if ( new_body_list == NULL ) return NULL;

	new_body_list->size = 0;
	new_body_list->first = NULL;
	new_body_list->last = NULL ;

	return new_body_list;
}

body_list_el* create_body_list_el( void )
{
  body_list_el* new_element = malloc(sizeof(body_list_el));
  if ( new_element == NULL ) return NULL;

  new_element->b = NULL;
  new_element->next = NULL;

  return new_element;
}

void add_body_last( body_list* bl, body* b )
{
  body_list_el* new_element;
  new_element = create_body_list_el();

  new_element->b = b;

  bl->last->next = new_element;
  bl->last = new_element;
  bl->size++;
}

void add_body_first( body_list* bl, body* b )
{
  body_list_el* new_element;
  new_element = create_body_list_el();

  new_element->next = bl->first;
  bl->first = new_element;
  bl->size++;
}

void free_body_list_el( body_list_el* element )
{
  if ( element->b != NULL ) free_body(element->b);
  free(element);
}

void free_body_list( body_list* bl )
{
  body_list_el* buff = bl->first;
  body_list_el* buff2;

  while ( buff != NULL )
  {
    buff2 = buff;
    buff = buff->next;
    free_body_list_el(buff2);
  }

  free(bl);
}
