#include <stdio.h>
#include "universe.h"

universe* global_universe;

void init( void )
{
	global_universe = create_universe( 500, 500 );
	generate_random_universe( global_universe );
}

void loop( void )
{
	while ( 1 ) /* todo : add a stop condition */ 
	{
		apply_strength_on_universe( global_universe );
	}
}

void end( void )
{
	free_universe( global_universe );
}

int main( void )
{
	init();
	loop();
	end();
}