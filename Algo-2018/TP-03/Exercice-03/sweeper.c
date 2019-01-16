/*
 * A mine sweeper using the MLV library
 */
#include <MLV/MLV_all.h>
#include "grid.h"
#include "draw.h"
#include <stdio.h>

/* Custom */
/*
#define GRID_WIDTH  36
#define GRID_HEIGHT 24
#define NUMBER_OF_MINES 10
*/

/* Beginner */
#define GRID_WIDTH  9
#define GRID_HEIGHT 9
#define NUMBER_OF_MINES 10

/* Intermediate */
/*
#define GRID_WIDTH  16
#define GRID_HEIGHT 16
#define NUMBER_OF_MINES 40
*/

/* Advanced */
/*
#define GRID_WIDTH  30
#define GRID_HEIGHT 16
#define NUMBER_OF_MINES 99
*/

int main() {

	/* Create the internal representation of the game board */
	grid *g = create_grid(GRID_WIDTH, GRID_HEIGHT);
	add_mines(g, NUMBER_OF_MINES);

	/* Create the window */
	int width=GRID_WIDTH*GRID_SCALE;
	int height=GRID_HEIGHT*GRID_SCALE;
	MLV_create_window("Sweeper", "Sweeper", width, height);

	/* Draw the game board in the window */
	/* set_all_visible(g); */
	draw_grid(g);
	MLV_actualise_window();


	/* Main game loop */
	MLV_Event event = MLV_NONE;
	MLV_Keyboard_button key_button;
	MLV_Mouse_button mouse_button;
	MLV_Button_state mouse_state;
	int x_pixel, y_pixel, x_pos, y_pos;
	int game_over = 0;
	int marked = 0;
	int remaining = GRID_WIDTH*GRID_HEIGHT-NUMBER_OF_MINES;

	while (!(event == MLV_KEY && key_button == MLV_KEYBOARD_ESCAPE)) {

		event = MLV_wait_event(
				&key_button,
				NULL,
				NULL,
				NULL,
				NULL,
				&x_pixel,
				&y_pixel,
				&mouse_button,
				&mouse_state
			);
		
		if (!game_over && event == MLV_MOUSE_BUTTON && mouse_state == MLV_RELEASED) {

			/* The player has pressed a button */
			x_pos = x_pixel/GRID_SCALE;
			y_pos = y_pixel/GRID_SCALE;
			printf("Mouse click on (%d,%d)\n", x_pos, y_pos);
			cell *c = &g->cells[x_pos][y_pos];

			if (mouse_button == MLV_BUTTON_LEFT && !c->visible && !c->marked) {

				/* The player has left clicked on an unmarked hidden cell */
				int uncovered = uncover(g,c);
				printf("Uncovered %d cell(s)\n", uncovered);
				remaining -= uncovered;
				if (c->mine) {
					/* The player has struck a mine */
					printf("Game over! Press ESCAPE to quit.\n");
					game_over = 1;
				}

			} else if (mouse_button == MLV_BUTTON_RIGHT && !c->visible) {

				/* The player has right clicked on a hidden cell */
				if (c->marked) {
					marked--;
					c->marked = 0;
				} else {
					marked++;
					c->marked = 1;
				}
				draw_cell_actualise_window(c);
			}

			if (remaining == 0 && marked == NUMBER_OF_MINES) {
					/* The player has marked all mines and
					   uncovered all other cells */
				printf("You win! Press ESCAPE to quit.\n");
				game_over = 1;
			}
		}
	}

	MLV_free_window();

	free_grid(g);

	return 0;
}