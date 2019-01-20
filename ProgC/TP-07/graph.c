#include <MLV/MLV_all.h>

#include "graph.h"

void initialize_window ( const char * name, const char * name2, int size_x, int size_y )
{
  MLV_create_window(name, name2, size_x, size_y);
}

void create_grid ( void )
{
  int i;

  for ( i = 0 ; i < 9 ; i++ )
  {
    MLV_draw_line(10+i*50, 10, 10+i*50, 450, MLV_COLOR_BLUE);
  }
}