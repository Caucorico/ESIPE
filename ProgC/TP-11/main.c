#include "board.h"
#include "window.h"

board* global_board;
image* global_image;

void init(void)
{
	global_board = initialize_board(global_board, 4, 4);
	global_image = init_image(global_image);
}

void loop(void)
{
	manage_events();
}

void end(void)
{

}

int main(int argc, char** argv)
{
	init();
	loop();
	end();
}