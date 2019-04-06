#ifndef __GRAPH__
#define __GRAPH__

#include "sudoku.h"

void initialize_window ( const char * name, const char * name2, int size_x, int size_y );

void create_grid ( void );

void start( Board grid, int tab[9][9] );

void loop( Board grid, int tab[9][9] );

#endif