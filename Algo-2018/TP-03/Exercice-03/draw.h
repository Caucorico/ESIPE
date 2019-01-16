#ifndef DRAW_H
#define DRAW_H
#include "grid.h"

#define GRID_SCALE 30

/*
 * Draw a single cell. Does *not* call MLV_actualise_window().
 */
void draw_cell(cell *c);

/*
 * Draw a single cell and call MLV_actualise_window().
 */
void draw_cell_actualise_window(cell *c);

/*
 * Draw the entire grid and call MLV_actualise_window() once.
 */
void draw_grid(grid *g);

#endif /* DRAW_H */