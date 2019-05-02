#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "board.h"

board* initialize_board(int nb_column, int nb_line)
{
	int i,j;
	board* new_board = malloc(sizeof(board));
	if ( new_board == NULL ) return NULL;

	new_board->nb_column = nb_column;
	new_board->nb_line = nb_line;

	for ( i = 0 ; i < nb_line ; i++ )
	{
		for ( j = 0 ; j < nb_column ; j++ )
		{
			new_board->block[i][j].line = i;
			new_board->block[i][j].column = j;
		}
	}

	new_board->empty_line = nb_line-1;
	new_board->empty_column = nb_column-1;

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
		if ( b->empty_column < b->nb_column-1 ) return (1);
		return (0);
	}
	else
	{
		if ( b->empty_line < b->nb_line-1 ) return (1);
		return(0);
	}
}

void swap_square( square* s1, square* s2 )
{
	square buff;
	buff = *s1;
	*s1 = *s2;
	*s2 = buff;
}

void move_square(board* b, unsigned char move)
{
	if ( move == 0 )
	{
		swap_square(&b->block[b->empty_line][b->empty_column], &b->block[b->empty_line][b->empty_column-1]);
		b->empty_column--;
	}
	else if ( move == 1 )
	{
		swap_square(&b->block[b->empty_line][b->empty_column], &b->block[b->empty_line-1][b->empty_column]);
		b->empty_line--;
	}
	else if ( move == 2 )
	{
		swap_square(&b->block[b->empty_line][b->empty_column], &b->block[b->empty_line][b->empty_column+1]);
		b->empty_column++;
	}
	else
	{
		swap_square(&b->block[b->empty_line][b->empty_column], &b->block[b->empty_line+1][b->empty_column]);
		b->empty_line++;
	}

}

unsigned char is_complete(board* b)
{
	int i, j;

	for ( i = 0 ; i < b->nb_line ; i++ )
	{
		for ( j = 0 ; j < b->nb_column ; j++ )
		{
			if ( b->block[i][j].line != i ||  b->block[i][j].column != j )
			{
				return 0;
			}
		}
	}

	return 1;
}

void display_ascii_board_on_stdout(board* b)
{
	int i,j;

	for ( i = 0 ; i < b->nb_line ; i++ )
	{
		for ( j = 0 ; j < b->nb_column ; j++ )
		{
			if ( i == b->empty_line && j == b->empty_column )
			{
				printf("|     ");
			}
			else
			{
				printf("| %d %d ", b->block[i][j].line, b->block[i][j].column);
			}
		}
		printf("|\n");
	}
	printf("###################################################\n");
}

void mix_board(board* b)
{
	int l, i, m;
	srand(time(NULL));

	l = rand()%255;

	for ( i = 0 ; i < l ; i++ )
	{
		m = rand()%4;
		if ( is_move_legal(b, m) )
		{
			move_square(b, m);
		}
	}
}