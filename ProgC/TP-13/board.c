#include <stdlib.h>
#include <stdio.h>
#include "board.h"
#include "bitboard.h"

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
	new_board->is_over = 0;

	for ( i = 0 ; i < 8 ; i++ )
	{
		new_board->queens[i] = -1;
	}

	return new_board;
}

void free_board( board* b )
{
	free(b);
}

void set_queen_attack(board* b, int index, void(*bit_action)(unsigned long int*, int))
{
	int i;

	for ( i = 0 ; i < 8 ; i++ )
	{
		bit_action(&b->bitboard, 63 - ((b->queens[index]*8)+i));
	}

	for ( i = 0 ; i < 8 ; i++ )
	{
		bit_action(&b->bitboard,  63 - ((8*i) + index) );
	}


	for ( i = 1 ; i <= b->queens[index] ; i++ )
	{

		if ( index-i >= 0 )
		{
			bit_action(&b->bitboard, 63-(((b->queens[index]*8)+index)-((i*8) + 1*i)) );
		}
	}

	for ( i = 1 ; i <= b->queens[index] ; i++ )
	{
		if ( index+i < 8 )
		{
			bit_action(&b->bitboard, 63-(((b->queens[index]*8)+index)-((i*8) - 1*i)) );
		}
	}

	for ( i = 1 ; i <= 7-b->queens[index] ; i++ )
	{
		if ( index+i < 8 )
		{
			bit_action(&b->bitboard, 63-(((b->queens[index]*8)+index)+((i*8) + 1*i)) );
		}
	}

	for ( i = 1 ; i <= 7-b->queens[index] ; i++ )
	{
		if ( index-i >= 0 )
		{
			bit_action(&b->bitboard, 63-(((b->queens[index]*8)+index)+((i*8) - 1*i)) );
		}
	}
}

void set_attacks(board* b, void(*bit_action)(unsigned long int*, int))
{
	int i;

	b->bitboard = 0;

	for ( i = 0 ; i < 8 ; i++ )
	{
		if ( b->queens[i] != -1 )
		{
			set_queen_attack(b, i, bit_action);
		}
	}
}

unsigned char is_attack(board* b, int x, int y)
{
	return bit_value_ULI(b->bitboard, 63-(y*8 + x));
}

unsigned char is_finish(board* b)
{
	if ( b->bitboard == 0xffffffffffffffff) return 1;
	return 0;
}

unsigned char is_win(board* b)
{
	int i;

	for ( i = 0 ; i < 8 ; i++ )
	{
		if ( b->queens[i] == -1 ) return 0;
	}
	return 1;
}