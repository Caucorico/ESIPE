#ifndef VISUALARRAY_H
#define VISUALARRAY_H
#include <MLV/MLV_all.h>

#define BACKGROUND_COLOR MLV_COLOR_BLACK
#define FOREGROUND_COLOR MLV_COLOR_WHITE

/**
 * Initialse the visualisation linked to the array arr, open a window,
 *   and visualise the array in the window.
 * The actual width of the window is size*(preffered_width/size).
 * The visualisation uses bars if bar != 0, and points if bar = 0.
 */
void init_visual(int *arr, int size, int preferred_width, int bar);

/**
 * Free memory and close the window. 
 * Does not free the memory used by the linked array.
 */
void free_visual();

/**
 * Visualise the entire array in the window.
 */
void visualize();

/**
 * Draw the positions pos[0], pos[1], ..., pos[n-1] using colour c.
 */
void visualize_positions(int pos[], int n, MLV_Color c);

/**
 * Draw the two positions i and j using colour c.
 */
void visualize_2_positions(int i, int j, MLV_Color c);

#endif /* VISUALARRAY_H */