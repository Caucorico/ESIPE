#ifndef _BOARD_
#define _BOARD_

typedef struct _board
{
	unsigned long int bitboard;
	char queens[8];
	unsigned int square_size;
	int x;
	int y;
  unsigned char is_over;
}board;

board* create_board( int x, int y, int square_size );

void free_board( board* b );

void set_queen_attack(board* b, int index, void(*bit_action)(unsigned long int*, int));

void set_attacks(board* b, void(*bit_action)(unsigned long int*, int));

/* Return true if the square is attacked */
unsigned char is_attack(board* b, int x, int y);

unsigned char is_finish(board* b);

unsigned char is_win(board* b);

#endif