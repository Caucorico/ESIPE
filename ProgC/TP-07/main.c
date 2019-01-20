#include <stdio.h>
#include "grapĥ.h"
#include "sudoku.h"
#include "in_out.h"

int main(int argc, char* argv[]){
  Board B;

  fread_board(argv[1], B);

  initialize_window("test", "test", 500, 500);



  return 0;
}
