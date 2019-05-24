#include <MLV/MLV_shape.h>
#include <MLV/MLV_text.h>
#include "graph_board.h"
#include "board.h"

void draw_checkerboard( board* b )
{
	int i, j;

	for ( i = 0 ; i < 8 ; i++ )
	{
		for ( j = 0 ; j < 8 ; j++ )
		{
			if ( (i*9 + j)%2 )
			{
				MLV_draw_filled_rectangle( b->x+(j*b->square_size), b->y+(i*b->square_size), b->square_size, b->square_size, MLV_COLOR_GRAY57 );
			}
			else
			{
				MLV_draw_filled_rectangle( b->x+(j*b->square_size), b->y+(i*b->square_size), b->square_size, b->square_size, MLV_COLOR_WHITE );
			}
		}
	}

	MLV_draw_rectangle(b->x, b->y, 8*b->square_size, 8*b->square_size, MLV_COLOR_BLACK);
}

void draw_numbers( board* b )
{
	int i;
	char number[2];
	number[1] = '\0';

	for ( i = 7 ; i >= 0 ; i-- )
	{
		number[0] = '0'+i;
		MLV_draw_text (b->x + b->square_size/2 + b->square_size * (7-i), b->y + b->square_size*8 +10, number, MLV_COLOR_BLACK );
	}

	for ( i = 7 ; i >= 0 ; i-- )
	{
		number[0] = '0'+i;
		MLV_draw_text (b->x + b->square_size*8 +10, b->y + b->square_size/2 + b->square_size * (7-i), number, MLV_COLOR_BLACK );
	}
}

void draw_queens( board* b )
{
	int i;

	for ( i = 0 ; i < 8 ; i++ )
	{
		if ( b->queens[i] != -1 )
		{
			MLV_draw_filled_circle( b->x + i*b->square_size + b->square_size/2, b->y + b->queens[i]*b->square_size + b->square_size/2, b->square_size/2, MLV_COLOR_GREEN );
		}
	}
}

void draw_board( board* b )
{
	draw_checkerboard( b );
	draw_numbers( b );
	draw_queens( b );
}