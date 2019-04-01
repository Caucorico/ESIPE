#ifndef __GRAPH__
#define __GRAPH__

#include "sudoku.h"

void initialize_window ( const char * name, const char * name2, int size_x, int size_y );

void create_grid ( void );

void start( Board grid );

void loop( Board grid );

#endif