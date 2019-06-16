#ifndef _WINDOW_
#define _WINDOW_

#include <MLV/MLV_image.h>
#include "board.h"

typedef struct _image 
{
	char* path;
	MLV_Image* mlv_im;
	int space;
}image;

image* init_image( const char* path );

void draw_image_square( image* im, int x, int y, int image_x, int image_y, int size_x, int size_y );

void draw_square( board* b, image* im, int line_x, int line_y );

void draw_board( board* b, image* im );

#endif