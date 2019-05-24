#include <stdlib.h>
#include <stdio.h>
#include "board.h"

board* create_board( int x, int y, int square_size )
{
	int i;
	board* new_board;

	new_board = malloc(sizeof(board));

	if ( new_board == NULL )
	{
		fprintf(stderr, "Error malloc in board.c in create_board\n");
		return NULL;
	}

	new_board->x = x;
	new_board->y = y;
	new_board->bitboard = 0;
	new_board->square_size = square_size;

	for ( i = 0 ; i < 8 ; i++ )
	{
		new_board->queens[i] = -1;
	}

	return new_board;
}