#include <MLV/MLV_all.h>

#define LINE_COLOR MLV_COLOR_BLUE
#define BACKGROUND_COLOR MLV_COLOR_WHITE

void draw_h(int x, int y, int width)
{
	if ( width < 8 ) return;

	/* trait droit du h*/
	MLV_draw_line(x-width/2, y, x+width/2, y, LINE_COLOR);

	/* les trait vericaux*/
	MLV_draw_line(x-width/2, y-width/2, x-width/2, y+width/2, LINE_COLOR);
	MLV_draw_line(x+width/2, y-width/2, x+width/2, y+width/2, LINE_COLOR);

	MLV_wait_milliseconds(10);
	MLV_actualise_window();

	/* recursion */
	draw_h(x-width/2,y-width/2, width/2);
	draw_h(x-width/2,y+width/2, width/2);
	draw_h(x+width/2,y-width/2, width/2);
	draw_h(x+width/2,y+width/2, width/2);

}

int main() {

	/* Create the window */
	MLV_create_window("Draw", "Draw", 500, 500);
	MLV_draw_filled_rectangle(0, 0, 499, 499, BACKGROUND_COLOR);

	draw_h(250, 250, 300);

	MLV_actualise_window();

	MLV_wait_seconds(5);

	MLV_free_window();

	return 0;
}