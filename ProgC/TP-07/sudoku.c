#include <stdio.h>

#include "sudoku.h"

void initialize_empty_board(Board grid){

}

void print_board(Board grid)
{
	int i,j;

	printf("\n");
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
	int i,j, k, l;
	int tmp[9];
	int board_size = (sizeof(Board)/sizeof(int)/9);

	for ( i = 0 ; i < board_size ; i++ )
	{
		for ( j = 0 ; j < board_size ; j++ )
		{
			if ( grid[j][i] == 0 ) continue;
			for ( k = j+1 ; k < board_size ; k++ )
			{
				if (grid[j][i] == grid[k][i]) return 0;
			}
		}

		for ( j = 0 ; j < board_size ; j++ )
		{
			if ( grid[i][j] == 0 ) continue;
			for ( k = j+1 ; k < board_size ; k++ )
			{
				if (grid[i][j] == grid[i][k]) return 0;
			}
		}
	}

	for ( i = 1 ; i < board_size ; i+=3 )
	{
		for ( j = 1 ; j < board_size ; j+=3 )
		{
			for ( k = 0 ; k < 9 ; k++ )
			{
				tmp[k] = 0;
			}
			for ( k = i-1 ; k <= i+1 ; k++ )
			{
				for ( l = j-1 ; l <= j+1 ; l++ )
				{
					if ( grid[k][l] != 0 )
					{
						tmp[grid[k][l]-1] += 1;
					}
				}
			}

			for ( k = 0 ; k < board_size ; k++ )
			{
				if ( tmp[k] > 1 )
				{
					return 0;
				}
			}
		}
	}

	return 1;
}

int board_solver(Board grid, int position, int max)
{
	int i,j,k,nbr=0;
	j = position/9;
	k = position%9;

	if ( position > max )
	{
		print_board(grid);
		return 1;
	}

	if ( grid[j][k] == 0)
	{
		for ( i = 1 ; i < 10 ; i++ )
		{
			grid[j][k] = i;
			if ( board_ok(grid) )
			{
				nbr += board_solver(grid, position+1, max);
			}
		}
		grid[j][k] = 0;
	}
	else
	{
		nbr += board_solver(grid, position+1, max);
	}
	return nbr;
}	

int board_finish(Board grid)
{
	int i,j;

	for ( i = 0 ; i < 9 ; i++ )
	{
		for ( j = 0 ; j < 9 ; j++ )
		{
			if ( grid[i][j] == 0 )
			{
				return 0;
			}
		}
	}

	return 1;
}