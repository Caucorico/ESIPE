#include "grid.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

/*
 * Allocate memory for a grid and initialize each cell.
 */
grid *create_grid(int x_size, int y_size) {
	int i, j;
	grid *g = (grid *)malloc(sizeof(grid));
	g->x_size = x_size;
	g->y_size = y_size;
	g->cells = (cell **)malloc(x_size*sizeof(cell *));
	for (i = 0; i < x_size; i++)
		g->cells[i] = (cell *)malloc(y_size*sizeof(cell));

	for (j = 0; j < y_size; j++)
		for (i = 0; i < x_size; i++) {
			g->cells[i][j].x_pos = i;
			g->cells[i][j].y_pos = j;
			g->cells[i][j].visible = 0;
			g->cells[i][j].marked = 0;
			g->cells[i][j].mine = 0;
			g->cells[i][j].mine_count = 0;
		}

	return g;
}

/*
 * Free memory for a grid.
 */
void free_grid(grid *g) {
	int i;
	for (i = 0; i < g->x_size; i++)
		free(g->cells[i]);
	free(g->cells);
	free(g);
}

/*
 * Set all cells to visible (for debugging).
 */
void set_all_visible(grid *g) {
	int x, y;
	for (x = 0; x < g->x_size; x++)
		for (y = 0; y < g->y_size; y++)
			g->cells[x][y].visible = 1;
}

/*
 * Add exactly n mines to grid g in random positions.
 */
void add_mines(grid *g, int n) {
	int i,j,k,x,y;

	srand(time(NULL));

	i = n;

	while ( i > 0 )
	{
		x = rand()%g->x_size;
		y = rand()%g->y_size;

		if ( g->cells[x][y].mine == 0 )
		{
			i--;
			g->cells[x][y].mine = 1;
			for ( j = x-1 ; j <= x+1 ; j++ )
			{
				for ( k = y-1 ; k <= y+1 ; k++ )
				{
					if ( j >= 0 && j < g->x_size && k >= 0 && k < g->y_size )
					{
						g->cells[j][k].mine_count += 1;
					}
				}
			}
		}
	}

	return;
}

/*
 * Uncover cell c in grid g.
 * Return the total number of cells uncovered.
 */
int uncover(grid *g, cell *c) {
	return 0;
}
