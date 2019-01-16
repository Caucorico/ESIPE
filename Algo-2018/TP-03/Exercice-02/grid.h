#ifndef GRID_H
#define GRID_H

/*
 * A single cell.
 *
 * int x_pos      : x coordinate of the cell in the grid.
 * int y_pos      : y coordinate of the cell in the grid.
 * int visible    : 1 if the cell is uncovered and 0 otherwise.
 * int marked     : 1 if the cell is marked for a mine and 0 otherwise.
 * int mine       : 1 if the cell contains a mine and 0 otherwise.
 * int mine_count : number of mines in adjacent cells (including itself).
 *
 */
typedef struct {
	int x_pos;
	int y_pos;
	int visible;
	int marked;
	int mine;
	int mine_count;
} cell;

/*
 * A grid of cells.
 *
 * cell **cells : a two-dimensional array of cells.
 * int x_size   : width of the grid (number of cells).
 * int y_size   : height of the grid (number of cells).
 *
 */
typedef struct {
	cell **cells;
	int x_size;
	int y_size;
} grid;

/*
 * Allocate memory for a grid and initialize each cell.
 */
grid *create_grid(int x_size, int y_size);

/*
 * Free memory for a grid.
 */
void free_grid(grid *g);

/*
 * Set all cells to visible (for debugging).
 */
void set_all_visible(grid *g);

/*
 * Add exactly n mines to grid g in random positions.
 */
void add_mines(grid *g, int n);

/*
 * Uncover cell c in grid g.
 * Return the total number of cells uncovered.
 */
int uncover(grid *g, cell *c);


#endif /* GRID_H */