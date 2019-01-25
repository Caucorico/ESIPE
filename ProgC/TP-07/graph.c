#include <MLV/MLV_all.h>

#include "graph.h"
#include "sudoku.h"

int determine_case(int v)
{
	return (v-10)/50;
}

int determine_selector_x(int x)
{
	return (x-500)/50;
}

int determine_selector_y(int y)
{
	return (y-150)/50;
}

int select_case(int v)
{
	return (v*50)+10;
}

void initialize_window ( const char * name, const char * name2, int size_x, int size_y )
{
  MLV_create_window(name, name2, size_x, size_y);
}

void create_grid ( void )
{
  int i;

  for ( i = 0 ; i < 10 ; i++ )
  {
    MLV_draw_line(10+i*50, 10, 10+i*50, 460, MLV_COLOR_BLUE);
  }

  for ( i = 0 ; i < 10 ; i++ )
  {
    MLV_draw_line(10, 10+i*50, 460, 10+i*50, MLV_COLOR_BLUE);
  }
}

void fill_grid ( Board grid )
{
	int i,j;
	char tab[2];
	tab[1] = '\0';

	for ( i = 0 ; i < 9 ; i++ )
	{
		for ( j = 0 ; j < 9 ; j++ )
		{
			if ( grid[j][i] != 0 )
			{
				tab[0] = '0'+grid[j][i];
				MLV_draw_text(20+select_case(i), 20+select_case(j), tab, MLV_COLOR_BLUE);
			}
		}
	}
}

void create_selector ( void )
{
	char i;
	char tab[2];
	tab[1] = '\0';

	for ( i = 0 ; i < 4 ; i++ )
	{
		MLV_draw_line(500+i*50, 160, 500+i*50, 310, MLV_COLOR_BLUE);
	}

	for ( i = 0 ; i < 4 ; i++ )
	{
		MLV_draw_line(500, 160+i*50, 650, 160+i*50, MLV_COLOR_BLUE);
	}

	for ( i = 0 ; i < 9 ; i++ )
	{
		tab[0] = i+'1';
		MLV_draw_text(525+(i%3)*50, 175+(i/3)*50, tab, MLV_COLOR_BLUE);
	}
}

void draw_char_in_case(int x, int y, char c)
{
	char tab[2];
	tab[0] = c;
	tab[1] = '\0';
	MLV_draw_filled_rectangle(select_case(x)+1, select_case(y)+1, 49, 49, MLV_COLOR_BLACK);
	MLV_draw_text(select_case(x)+20, select_case(y)+20, tab, MLV_COLOR_BLUE);
	MLV_actualise_window();
}

void start( Board grid )
{
	initialize_window("test", "test", 750, 500);
	create_grid();
	fill_grid ( grid );
	create_selector();
	MLV_actualise_window();
}

void loop( Board grid )
{
	int x,y;
	int x2, y2;
	int state=0;

	while( !board_finish(grid) || !board_ok(grid) )
	{
		MLV_wait_mouse(&x, &y);
		if ( x > 10 && x < 460 && y > 10 && y < 460 && state == 0)
		{
			x2 = determine_case(x);
			y2 = determine_case(y);

			draw_char_in_case(x2,y2,'?');

			state = 1;

		}
		if ( x > 500 && x < 650 && y > 150 && y < 300 && state == 1 )
		{
			printf("test : %d\n", determine_selector_y(y)*3 + 1+determine_selector_x(x));
			grid[y2][x2] = determine_selector_y(y)*3 + 1+determine_selector_x(x);
			draw_char_in_case(x2,y2,'0'+determine_selector_y(y)*3 + 1+determine_selector_x(x));
			MLV_actualise_window();

			state = 0;
		}

		printf(" finish ? : %d\n", board_finish(grid));
		printf("ok ? %d\n", board_ok(grid));
		
	}

}