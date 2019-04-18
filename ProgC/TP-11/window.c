#include <string.h>
#include <MLV_shape.h>
#include "window.h"
#include "board.h"

image* init_image( const char* path )
{
	image* new_image;
	char* p;

	new_image = malloc(sizeof(image));
	if ( new_image == NULL ) return NULL;

	p = malloc( sizeof(char) * ( strlen(path) + 1 ) );
	if ( p == NULL )
	{
		free(new_image);
		return NULL;
	}

	strcpy( new_image->path, path );
	new_image->mlv_im = MLV_load_image(path);
}

void free_image( image* im )
{
	if ( im != NULL )
	{
		if ( im->path != NULL ) free( im->path );
		if ( im->mlv_im != NULL ) free( im->mlv_im );
		free(im);
	}
}

void draw_image_square( image* im, int x, int y, int image_x, int image_y int size_x, int size_y )
{
	int i, j;
	int r, g, b, a;
	MLV_Color color;

	for ( i = 0 ; i < size_y ; i++ )
	{
		for ( j = 0 ; j < size_x ; j++ )
		{
			MLV_get_pixel_on_image( im->mlv_im, image_x+j, image_y+i, &r, &g ,&b, &a );
			color = MLV_rgba(r, g, b, a);
			MLV_draw_pixel( x+j, y+i, color );
		}
	}
}

void draw_square( board* board, image* im, int line_x, int line_y )
{
	int x, y;
	int size_x, size_y;
	int image_x, image_y;

	x = ((line_x+1)*im->space)+(line_x*(MLV_get_image_height(im->mlv_im)/board->nb_column));
	y = ((line_y+1)*im->space)+(line_y*(MLV_get_image_height(im->mlv_im)/board->nb_line));
	size_x = MLV_get_image_height(im->mlv_im)/board->nb_column;
	size_y = MLV_get_image_height(im->mlv_im)/board->nb_line;
	image_x = size_x*line_x;
	image_y = size_y*line_y;

	draw_image_square( im, x, y, image_x, image_y, size_x, size_y );
}