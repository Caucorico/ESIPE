#ifndef _EVENT_
#define _EVENT_

#include <MLV/MLV_event.h>
#include "board.h"

void click_on_checkboard( board* b, int mouse_x, int mouse_y );

void on_click_event( board* b, MLV_Mouse_button mouse_button, MLV_Button_state button_state, int mouse_x, int mouse_y );

void treat_event( board* b );

#endif