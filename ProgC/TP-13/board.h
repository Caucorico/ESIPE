#ifndef _BOARD_
#define _BOARD_

typedef struct _board
{
	unsigned long int bitboard;
	char queens[8];
	unsigned int square_size;
	int x;
	int y;
}board;

board* create_board( int x, int y, int square_size );

void set_attacks(board* b);

#endif