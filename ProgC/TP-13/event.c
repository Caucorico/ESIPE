#include <MLV/MLV_event.h>
#include "event.h"
#include "bitboard.h"

unsigned char is_click_on_board( board* b, int mouse_x, int mouse_y )
{
	unsigned int bl = 0;

	bl = ( mouse_x > b->x ) && ( (unsigned int)mouse_x < (b->x + 8*b->square_size) );
	bl &= ( mouse_y > b->y ) && ( (unsigned int)mouse_y < (b->y + 8*b->square_size) );

	return bl;
}

int get_square_x( board* b, int x )
{
	return ( x - b->x )/b->square_size;
}

int get_square_y( board* b, int y )
{
	return ( y - b->y )/b->square_size;
}

unsigned char is_click_on_exit(board* b, int mouse_x, int mouse_y)
{
	if ( (unsigned int)mouse_x > b->x + b->square_size*3 + b->square_size/2 && (unsigned int)mouse_y > b->y + (b->square_size*9)
		&& (unsigned int)mouse_y < b->y + (b->square_size*9) + 35 && (unsigned int)mouse_x < b->x + b->square_size*3 + b->square_size/2 + 115 )
		return 1;
	return 0;
}

void left_click_on_checkboard( board* b, int mouse_x, int mouse_y )
{
	if ( is_click_on_board(b, mouse_x, mouse_y) 
		&& b->queens[get_square_x(b, mouse_x)] != get_square_y(b, mouse_y) 
		&& !is_attack(b,get_square_x(b, mouse_x), get_square_y(b, mouse_y)) )
	{
		/*set_queen_attack(b, get_square_x(b, mouse_x), set_negative_bit_ULI);*/
		b->queens[get_square_x(b, mouse_x)] = get_square_y(b, mouse_y);
		set_attacks(b, set_positive_bit_ULI);
	}
	else if ( is_click_on_exit(b, mouse_x, mouse_y) )
	{
		b->is_over = 1;
	}
}

void right_click_on_checkboard( board* b, int mouse_x, int mouse_y )
{
	if ( is_click_on_board(b, mouse_x, mouse_y) && b->queens[get_square_x(b, mouse_x)] == get_square_y(b, mouse_y) )
	{
		b->queens[get_square_x(b, mouse_x)] = -1;
		set_attacks(b, set_positive_bit_ULI);
	}
}

void on_click_event( board* b, MLV_Mouse_button mouse_button, MLV_Button_state button_state, int mouse_x, int mouse_y )
{
	if ( mouse_button == MLV_BUTTON_LEFT && button_state == MLV_PRESSED )
	{
		left_click_on_checkboard(b, mouse_x, mouse_y);
	}
	else if ( mouse_button == MLV_BUTTON_RIGHT && button_state == MLV_PRESSED )
	{
		right_click_on_checkboard(b, mouse_x, mouse_y);
	}
}

void treat_event( board* b )
{
	int mouse_x, mouse_y;
	MLV_Mouse_button mouse_button;
	MLV_Button_state button_state;

	MLV_wait_event( NULL, NULL, NULL, NULL, NULL, &mouse_x, &mouse_y, &mouse_button, &button_state );

	if ( mouse_button == MLV_BUTTON_LEFT || mouse_button == MLV_BUTTON_MIDDLE || mouse_button == MLV_BUTTON_RIGHT )
	{
		on_click_event(b, mouse_button, button_state, mouse_x, mouse_y);
	}
}