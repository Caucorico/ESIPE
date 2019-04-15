#ifndef _UNIVERSE_
#define _UNIVERSE_

typedef struct _universe
{
	int size_x;
	int size_y;
	body_list* bl;
}universe;

/* create an empty universe */
universe* create_universe( int size_x, int size_y );

/* Free the universe and its content */
void free_universe( universe* u );

/* create a random universe on u */
void generate_random_universe( universe* u );

void apply_strength_on_universe( universe* u );

#endif

