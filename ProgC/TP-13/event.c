#include <MLV/MLV_event.h>
#include "event.h"

void click_on_checkboard( int mouse_x, int mouse_y )
{

}

void on_click_event( MLV_Mouse_button mouse_button, MLV_Button_state button_state, int mouse_x, int mouse_y )
{
	if ( mouse_button == MLV_BUTTON_LEFT && button_state == MLV_RELEASED )
	{

	}
}

void treat_event( void )
{
	int mouse_x, mouse_y;
	MLV_Mouse_button mouse_button;
	MLV_Button_state button_state;

	MLV_wait_event( NULL, NULL, NULL, NULL, NULL, &mouse_x, &mouse_y, &mouse_button, &button_state );

	if ( mouse_button == MLV_BUTTON_LEFT || mouse_button == MLV_BUTTON_MIDDLE || mouse_button == MLV_BUTTON_MIDDLE )
	{
		on_click_event(mouse_button, button_state, mouse_x, mouse_y);
	}
}