#ifndef __SUDOKU__
#define __SUDOKU__

typedef int Board[9][9];

void initialize_empty_board(Board grid);
void print_board(Board grid);
int board_ok(Board grid);
void board_solver(Board grid, int current, int max);

#endif
