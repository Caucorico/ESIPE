#include <MLV/MLV_window.h>
#include <stdio.h>
#include "bitboard.h"
#include "graph_board.h"
#include "board.h"
#include "event.h"

board* global_board;

void init( void )
{
	global_board = create_board(10, 10, 75);
	MLV_create_window("tp-13", "tp-13", 1000, 1000);
	MLV_clear_window(MLV_COLOR_WHITE);
	draw_board(global_board);
	MLV_actualise_window();
}

void loop( void )
{
	while ( 1 )
	{
		treat_event( global_board );
		if ( isCompromised( global_board ) )
		{
			printf("argh !\n");
		}
		MLV_clear_window(MLV_COLOR_WHITE);
		draw_board(global_board);
		MLV_actualise_window();
	}
}

void end ( void )
{

}

int main(void)
{

	init();

	loop();


	return 0;
}