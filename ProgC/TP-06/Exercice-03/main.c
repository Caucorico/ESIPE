#include <stdio.h>

#include "sudoku.h"
#include "in_out.h"

int main(int argc, char* argv[]){
  Board B;
  int i;

  printf("########################################\n");
  printf("TP-06 Exercice-03. \nBut : Trouver le ou les solutions d'une grille de sudoku. \n\n");

  /* Zone TP */

  fread_board(argv[1], B);

  i = board_solver(B, 0, 80);

  printf("result = %d", i);

  /* Fin zone TP */

  printf("\n\n########################################\n");
  return 0;
}
