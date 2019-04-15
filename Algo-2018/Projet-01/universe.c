#include "universe.h"

universe* create_universe( int size_x, int size_y )
{
	universe* new_universe = malloc(sizeof(universe));
	if ( new_universe == NULL ) return NULL;

	new_universe->size_x = size_x;
	new_universe->size_y = size_y;
	new_universe->bl = NULL;

	return universe;
}