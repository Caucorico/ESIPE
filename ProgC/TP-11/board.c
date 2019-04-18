#include "board.h"

board* initialize_board(int nb_column, int nb_line)
{
	int i,j;
	board* new_board = malloc(sizeof(board));
	if ( new_board == NULL ) return NULL;

	for ( i = 0 ; i < nb_line ; i++ )
	{
		for ( j = 0 ; j < nb_column ; j++ )
		{
			new_board->block[i][j].line = i;
			new_board->block[i][j].column = j;
		}
	}

	b->empty = &block[nb_line-1][nb_column-1];

	return new_board;
}

unsigned char is_move_legal(board* b, unsigned char move)
{
	if ( move == 0 )
	{
		if ( b->empty_column > 0 ) return (1);
		return (0);
	}
	else if ( move == 1 )
	{
		if ( b->empty_line > 0 ) return (1);
		return (0);
	}
	else if ( move == 2 )
	{
		if ( b->empty_column < NB_COL-1 ) return (1);
		return (0);
	}
	else
	{
		if ( b->empty_line < NB_LINE-1 ) return (1);
		return(0);
	}
}

void swap_square( square* s1, square* s2 )
{
	square buff;
	buff = s1;
	*s1 = *s2;
	*s2 = buff;
}

void move_square(board* b, unsigned char move)
{
	if ( move == 0 )
	{
		swap_square(&b->block[b->empty_line][b->empty_column], &b->block[b->empty_line][b->empty_column-1]);
	}
	else if ( move == 1 )
	{
		swap_square(&b->block[b->empty_line][b->empty_column], &b->block[b->empty_line-1][b->empty_column]);
	}
	else if ( move == 2 )
	{
		swap_square(&b->block[b->empty_line][b->empty_column], &b->block[b->empty_line][b->empty_column+1]);
	}
	else
	{
		swap_square(&b->block[b->empty_line][b->empty_column], &b->block[b->empty_line+1][b->empty_column]);
	}
}