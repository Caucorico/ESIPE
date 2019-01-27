#include <stdio.h>
#include "sudoku.h"
#include "in_out.h"
#include "graph.h"

int main(int argc, char* argv[]){
  Board B;
  int tab[9][9];

  fread_board(argv[1], B);

  start(B, tab);
  loop(B, tab);


  return 0;
}
