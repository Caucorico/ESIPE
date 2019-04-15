#include "body_list.h"
#include "body.h"

bodies_list* create_body_list( void )
{
	body_list* new_body_list = malloc(sizeof(body_list));
	if ( new_body_list == NULL ) return NULL;

	new_body_list->size = 0;
	new_body_list->first = NULL;
	new_body_list->last = NULL ;

	return new_body_list;
}

void add_body_last( body_list* bl, body* b )
{
	bl->last->next = b;
	bl->last = b;
}

void add_body_first( body_list* bl, body* b )
{
	b->next = bl->first;
	bl->first = b;
}

void free_body_list( body_list* bl )
{
	int i;
	body* buff;
	body* buff2;

	buff = bl->first;
	while ( buff != NULL )
	{
		buff2 = buff;
		buff = buff->next;
		free(buff2);
	}

	free(bl);
}