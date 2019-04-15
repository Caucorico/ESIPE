#ifndef _BODY_LIST_
#define _BODY_LIST_

typedef struct _body_list
{
	int size;
	body* first;
	body* last;
}body_list;


/* return an empty body list */
body_list* create_body_list( void );

/* add a body at the end of the list */
void add_body_last( body_list* bl, body* b );

/* add a body at the begening of the list */
void add_body_first( body_list* bl, body* b );

/* free the list and its content */
void free_body_list( body_list* bl );

#endif