#include <stdlib.h>
#include <stdio.h>
#include "event.h"

typedef struct _link {
    event* e;
    struct _link* next;
} link;

typedef struct _prioqueue {
    link* first;
    int size;
} prioqueue;

prioqueue *create_pq()
{
    prioqueue* new_pq = (prioqueue*)malloc(sizeof(prioqueue));
    if ( new_pq == NULL ) return NULL;
    new_pq->first = NULL;
    new_pq->size = 0;
    return new_pq;
}

void free_pq(prioqueue *q)
{
    if ( q->first != NULL ) printf("argh\n");
    free(q);
}

int size_pq(prioqueue *q)
{
    if ( q != NULL )
        return q->size;
    else
        return -1; 
}

void insert_pq(prioqueue *q, event *e)
{
	link* new_link = (link*)malloc(sizeof(link));
	if(new_link != NULL){
		new_link->e = e;
		if(q->first == NULL){
			new_link->next = NULL;
			q->first = new_link;
			q->size++;
		}
		else{
			link* empty_link = q->first;
			q->size++;

			if(empty_link->e->etime >= e->etime){
				new_link->next = empty_link;
				q->first = new_link;
			}
			else{
				while(empty_link->next != NULL && empty_link->next->e->etime < e->etime){
					empty_link = empty_link->next;
				}

				new_link->next = empty_link->next;
				empty_link->next = new_link;
			}
		}
	}
}

void display_pq(prioqueue *pq)
{
    link* buff = pq->first;
    while ( buff != NULL )
    {
        printf("%d~", buff->e->etime);
        buff = buff->next;
    }
    putchar('\n');
}

event* remove_min_pq(prioqueue *q)
{
    link* buff = q->first;
    event* ev;
    if ( buff == NULL ) return NULL;
    q->first = buff->next;
    q->size--;
    ev = buff->e;
    free(buff);
    return ev;
}
