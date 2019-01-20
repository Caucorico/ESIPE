#include <stdio.h>

#include "sudoku.h"
#include "in_out.h"

int main(int argc, char* argv[]){
  Board B;
  int i;

  fread_board(argv[1], B);

  i = board_solver(B, 0, 80);

  printf("result = %d", i);

  return 0;
}
