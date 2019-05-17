#include <MLV/MLV_window.h>
#include <stdio.h>
#include "bitboard.h"
#include "graph_board.h"


void init( void )
{
	MLV_create_window("tp-13", "tp-13", 1000, 1000);
	MLV_clear_window(MLV_COLOR_WHITE);
	draw_board(10, 10, 75 );
	MLV_actualise_window();
}

void loop( void )
{

}

void end ( void )
{

}

int main(void)
{
	unsigned long int test;
	int i;

	init();

	test = 0x0;

	for ( i = 0 ; i < 64 ; i++ )
	{
		set_positive_bit_ULI(&test, i);
		print_ULI(test);
		set_negative_bit_ULI(&test, i);
		scanf("%*c");
		putchar('\n');
	}


	return 0;
}