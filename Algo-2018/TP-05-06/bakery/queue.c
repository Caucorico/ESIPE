#include <stdlib.h>
#include <stdio.h>
#include "queue.h"

typedef struct _link {
    customer*       c;
    struct _link*   next;
} link;

struct _queue {
    link*   first;
    link*   last;
    int     size;
};

queue *create_q() {
    queue *q = (queue*)malloc(sizeof(queue));
    q->first = NULL;
    q->last = NULL;
    q->size = 0;
    return q;
}

void free_q(queue *q) {
    free(q);
}

int size_q(queue *q) {
    return q->size;
}

void enqueue_q(queue *q, customer *c)
{
    link* buff;
    link* new_link;

    new_link = (link*)malloc(sizeof(link));
    if ( new_link != NULL )
    {
        new_link->c = c;
        new_link->next = NULL;

        /* this cond work only if the queue is always coherent */
        if ( q->first == NULL && q->last == NULL )
        {
            q->first = new_link;
            q->last = new_link;
            q->size++;
        }
        else
        {
            buff = q->last;
            buff->next = new_link;
            q->last = new_link;
            q->size++;
        }
    }
}

void display_q(queue *q)
{
    link* buff = q->first;
    while( buff != NULL )
    {
        putchar('X');
        buff = buff->next;
    }
    putchar('\n');
}

customer* dequeue_q(queue* q)
{
    link* buff = q->first;
    customer* cust;
    if ( buff == NULL ) return NULL;
    q->first = buff->next;
    if ( q->size == 1 ) q->last = NULL;
    q->size--;
    cust = buff->c;
    free(buff);
    return cust;
}