#include <stdio.h>
#include "sudoku.h"
#include "in_out.h"
#include "graph.h"

int main(int argc, char* argv[]){
  Board B;

  fread_board(argv[1], B);

  start(B);
  loop(B);


  return 0;
}
