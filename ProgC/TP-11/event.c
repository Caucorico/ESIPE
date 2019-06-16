#include <MLV/MLV_event.h>
#include "event.h"
#include "board.h"

void manage_events( board* b )
{
	MLV_Keyboard_button kb;
	MLV_Button_state state;

	MLV_wait_event(&kb, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &state);

  if ( state == MLV_PRESSED )
  {
    if ( kb == MLV_KEYBOARD_LEFT )
    {
      if ( is_move_legal(b, 0) )
      {
        move_square(b, 0);
      }
    }

    if ( kb == MLV_KEYBOARD_UP )
    {
      if ( is_move_legal(b, 1) )
      {
        move_square(b, 1);
      }
    }

    if ( kb == MLV_KEYBOARD_RIGHT )
    {
      if ( is_move_legal(b, 2) )
      {
        move_square(b, 2);
      }
    }

    if ( kb == MLV_KEYBOARD_DOWN )
    {
      if ( is_move_legal(b, 3) )
      {
        move_square(b, 3);
      }
    }

  }
}