#include <MLV/MLV_all.h>
#include "draw.h"

/*
 * Draw a filled rectangle in position (x,y) using fill_color with a border in line_color.
 */
void draw_rectangle(int x, int y, MLV_Color line_color, MLV_Color fill_color) {
	MLV_draw_filled_rectangle(
			GRID_SCALE*x,
			GRID_SCALE*y,
			GRID_SCALE,
			GRID_SCALE,
			fill_color
		);
	MLV_draw_rectangle(
			GRID_SCALE*x,
			GRID_SCALE*y,
			GRID_SCALE,
			GRID_SCALE,
			line_color
		);
}

/*
 * Draw a single cell. Does *not* call MLV_actualise_window().
 */
void draw_cell(cell *c) {
	char label[] = "0";
	MLV_Color col;
	if (c->marked) {
		draw_rectangle(c->x_pos, c->y_pos, MLV_COLOR_GRAY50, MLV_COLOR_BLACK);
	} else if (!c->visible) {
		draw_rectangle(c->x_pos, c->y_pos, MLV_COLOR_GRAY50, MLV_COLOR_GRAY70);
	} else if (c->visible && c->mine) {
		draw_rectangle(c->x_pos, c->y_pos, MLV_COLOR_GRAY50, MLV_COLOR_RED);
	} else if (c->visible && !c->mine) {
		draw_rectangle(c->x_pos, c->y_pos, MLV_COLOR_GRAY50, MLV_COLOR_GRAY60);
		if (c->mine_count > 0) {
			sprintf(label, "%c", '0'+c->mine_count);
			switch(c->mine_count) {
				case 1: col = MLV_COLOR_BLUE; break;
				case 2: col = MLV_COLOR_GREEN; break;
				case 3: col = MLV_COLOR_RED; break;
				case 4: col = MLV_COLOR_DARK_BLUE; break;
				case 5: col = MLV_COLOR_BROWN; break;
				case 6: col = MLV_COLOR_CYAN; break;
				case 7: col = MLV_COLOR_BLACK; break;
				case 8: col = MLV_COLOR_GRAY; break;
				default: col = MLV_COLOR_YELLOW;
			}
			MLV_draw_text(
					GRID_SCALE*c->x_pos+(GRID_SCALE-10)/2,
					GRID_SCALE*c->y_pos+(GRID_SCALE-10)/2,
					label,
					col
				);
		}
	}
}

/*
 * Draw a single cell and call MLV_actualise_window().
 */
void draw_cell_actualise_window(cell *c) {
	draw_cell(c);
	MLV_actualise_window();
}

/*
 * Draw the entire grid and call MLV_actualise_window() once.
 */
void draw_grid(grid *g) {
	int i, j;
	for (i = 0; i < g->x_size; i++)
		for (j = 0; j < g->y_size; j++)
			draw_cell(&g->cells[i][j]);
	MLV_actualise_window();
}