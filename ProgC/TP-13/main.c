#include <MLV/MLV_window.h>
#include <stdio.h>
#include "bitboard.h"
#include "graph_board.h"
#include "board.h"
#include "event.h"

void end ( void* data )
{
	board* b = (board*)data;
	free_board(b);
	MLV_free_window();
}

board* init( void )
{
	board* b;
	b = create_board(10, 10, 75);
	MLV_execute_at_exit(end, (void*)b); /* segfault */
	MLV_create_window("tp-13", "tp-13", 1000, 1000);
	MLV_clear_window(MLV_COLOR_WHITE);
	draw_board(b);
	MLV_actualise_window();

	return b;
}

void loop( board* b )
{
	while ( b->is_over != 1 ) /* I use a flag :( */
	{
		treat_event( b );
		MLV_clear_window(MLV_COLOR_WHITE);
		draw_board(b);
		MLV_actualise_window();
	}
}

int main(void)
{

	board* b;

	b = init();

	loop(b);

	end( b );


	return 0;
}