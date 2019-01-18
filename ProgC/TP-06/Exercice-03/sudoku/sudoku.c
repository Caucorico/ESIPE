#include <stdio.h>

#include "sudoku.h"

void initialize_empty_board(Board grid){

}

void print_board(Board grid)
{
	int i,j;

	printf("\n");

	printf("%lu\n", sizeof(Board));
	for ( i = 0 ; i < (sizeof(Board)/sizeof(int)/9) ; i++ )
	{
		for ( j = 0 ; j < ((sizeof(Board)/sizeof(int)/9)*4)+1 ; j++)
		{
			printf("-");
		}
		printf("\n|");
		for ( j = 0 ; j < (sizeof(Board)/sizeof(int)/9) ; j++ )
		{
			printf(" %d |", grid[i][j]);
		}
		printf("\n");
	}
	for ( j = 0 ; j < ((sizeof(Board)/sizeof(int)/9)*4)+1 ; j++)
	{
		printf("-");
	}
	printf("\n");
}

int board_ok(Board grid)
{
	int i,j;
	int board_size = (sizeof(Board)/sizeof(int)/9);

	for ( i = 0 ; i < board_size ; i++ )
	{
		
	}
}