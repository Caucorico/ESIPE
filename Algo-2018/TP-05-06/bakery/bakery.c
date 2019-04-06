#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "event.h"
#include "customer.h"
#include "queue.h"
#include "prioqueue.h"

#define N_VENDORS 3
#define ARRIVAL_RATE (1.0/60)
#define MEAN_SERVICE_TIME 150

prioqueue*  event_queue;
queue*      customer_queue;
customer*   vendor[N_VENDORS];
int current_time;

double normal_delay( double mean )
{
	return -mean*log(1-((double)rand()/RAND_MAX));
}

void display_state( void )
{
	int i;

	printf("%3d | ", current_time);
	for ( i = 0 ; i < N_VENDORS ; i++ )
	{
		if ( vendor[i] == NULL ) putchar('_');
		else putchar('X');
	}
	printf(" | ");
	for ( i = 0 ; i < size_q(customer_queue) ; i++ )
	{
		putchar('X');
	}
	putchar('\n');
}

void add_customer(customer* c)
{
	int i;
	event* new_e;
	for ( i = 0 ; i < N_VENDORS ; i++ )
	{
		if ( vendor[i] == NULL )
		{
			vendor[i] = c;
			new_e = create_departure( current_time+150, c );
			insert_pq(event_queue, new_e);
			return;
		}
	}

	enqueue_q(customer_queue, c);

}

void remove_customer(customer* c)
{
	int i=0;
	customer* c2;
	event* new_e;

	while( vendor[i] != c ) i++;
	vendor[i] = NULL;
	free_customer(c);

	if ( size_q(customer_queue) > 0 )
	{
		c2 = dequeue_q(customer_queue);
		vendor[i] = c2;
		new_e = create_departure( current_time+normal_delay(MEAN_SERVICE_TIME), c2 );
		insert_pq(event_queue, new_e);
	}
}

void process_arrival(event *e)
{
	int time;
	event* new_e;
	customer* new_c;

	add_customer(e->c);

	time = current_time+normal_delay(1.0/ARRIVAL_RATE);
	new_c = create_customer(time);

	new_e = create_arrival( time, new_c );
	insert_pq(event_queue, new_e);

}

void process_departure(event *e)
{
	remove_customer(e->c);
}

void init_simu( void )
{
	int i;

	event_queue = create_pq();

	customer_queue = create_q();

	for ( i = 0 ; i < N_VENDORS ; i++ )
	{
		vendor[i] = NULL;
	}

	current_time = 0;
}

void treat_event_queue( void )
{
	event* current_event;
	while ( size_pq(event_queue) > 0 && CLOSING_TIME > current_time )
	{

		current_event = remove_min_pq(event_queue);
		current_time = current_event->etime;
		if ( current_event->type == EVENT_ARRIVAL )
		{
			process_arrival( current_event );	
		}
		else if ( current_event->type == EVENT_DEPARTURE )
		{
			process_departure( current_event );
		}
		display_state();
	}

}

int main() {
	customer* c;
	event* e;
	/*srand(time(NULL));*/

	init_simu();

	c = create_customer(42);
	e = create_arrival(42, c);
	insert_pq(event_queue, e);

	/*printf("debug");*/

	treat_event_queue();

	free_customer(c);
	

    return 0;
}