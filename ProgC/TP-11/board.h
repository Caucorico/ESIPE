#ifndef _BOARD_
#define _BOARD_

typedef struct _square
{
	int line;
	int column;
}square;

typedef struct _board
{
	square block[4][4];
	int empty_line;
	int empty_column;
	int nb_column;
	int nb_line;
}board;

board* initialize_board(int nb_column, int nb_line);

unsigned char is_move_legal(board* b, unsigned char move);

void move_square(board* b, unsigned char move);

unsigned char is_complete(board* b);

void display_ascii_board_on_stdout(board* b);

#endif