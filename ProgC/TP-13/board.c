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

	for ( i = 0 ; i < 8 ; i++ )
	{
		new_board->queens[i] = -1;
	}

	return new_board;
}

void set_queen_attack(board* b, int index)
{
	int i;

	if ( b->queens[index] != -1 )
	{
		for ( i = 0 ; i < 8 ; i++ )
	{
		if ( index != i && b->queens[index] != -1 )
		{
			set_positive_bit_ULI(&b->bitboard, 63 - ((b->queens[index]*8)+i));
		}
	}

	for ( i = 0 ; i < 8 ; i++ )
	{
		if ( b->queens[index] != i && b->queens[index] != -1 )
		{
			set_positive_bit_ULI(&b->bitboard,  63 - ((8*i) + index) );
		}
	}


	for ( i = 1 ; i <= b->queens[index] ; i++ )
	{

		if ( index-i >= 0 )
		{
			set_positive_bit_ULI(&b->bitboard, 63-(((b->queens[index]*8)+index)-((i*8) + 1*i)) );
		}
	}

	for ( i = 1 ; i <= b->queens[index] ; i++ )
	{
		if ( index+i < 8 )
		{
			set_positive_bit_ULI(&b->bitboard, 63-(((b->queens[index]*8)+index)-((i*8) - 1*i)) );
		}
	}

	for ( i = 1 ; i <= 7-b->queens[index] ; i++ )
	{
		if ( index+i < 8 )
		{
			set_positive_bit_ULI(&b->bitboard, 63-(((b->queens[index]*8)+index)+((i*8) + 1*i)) );
		}
	}

	for ( i = 1 ; i <= 7-b->queens[index] ; i++ )
	{
		if ( index-i >= 0 )
		{
			set_positive_bit_ULI(&b->bitboard, 63-(((b->queens[index]*8)+index)+((i*8) - 1*i)) );
		}
	}
	}

}

void set_attacks(board* b)
{
	int i;

	for ( i = 0 ; i < 8 ; i++ )
	{
		set_queen_attack(b, i);
	}
	print_ULI(b->bitboard);
}

unsigned char isCompromised(board* b)
{
	int i;

	for ( i = 0 ; i < 8 ; i++ )
	{
		if ( b->queens[i] != -1 && bit_value_ULI(b->bitboard, 63-(b->queens[i]*8 + i)) == 1 )
		{
				return 1;
		}
	}

	return 0;
}