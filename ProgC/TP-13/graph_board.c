#include <MLV/MLV_shape.h>
#include <MLV/MLV_text.h>
#include "graph_board.h"

void draw_checkerboard( int x, int y, int square_size )
{
	int i, j;

	for ( i = 0 ; i < 8 ; i++ )
	{
		for ( j = 0 ; j < 8 ; j++ )
		{
			if ( (i*9 + j)%2 )
			{
				MLV_draw_filled_rectangle( x+(j*square_size), y+(i*square_size), square_size, square_size, MLV_COLOR_GRAY57 );
			}
			else
			{
				MLV_draw_filled_rectangle( x+(j*square_size), y+(i*square_size), square_size, square_size, MLV_COLOR_WHITE );
			}
		}
	}

	MLV_draw_rectangle(x, y, 8*square_size, 8*square_size, MLV_COLOR_BLACK);
}

void draw_numbers( int x, int y, int square_size )
{
	int i;
	char number[2];
	number[1] = '\0';

	for ( i = 7 ; i >= 0 ; i-- )
	{
		number[0] = '0'+i;
		MLV_draw_text (x + square_size/2 + square_size * (7-i), y + square_size*8 +10, number, MLV_COLOR_BLACK );
	}

	for ( i = 7 ; i >= 0 ; i-- )
	{
		number[0] = '0'+i;
		MLV_draw_text (x + square_size*8 +10, y + square_size/2 + square_size * (7-i), number, MLV_COLOR_BLACK );
	}
}

void draw_board( int x, int y, int square_size )
{
	draw_checkerboard( x, y, square_size );
	draw_numbers( x, y, square_size );
}