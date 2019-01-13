#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "stack.h"
#include "interface.h"
#include "operation.h"

int main(int argc, char* argv[]){
  char str_action[256];
  int action=-1;

  stack_init();

  create_windows();
  display_stack();
  actualise_window();
  while (action != 0){
    get_user_action(str_action);
    action = analyse_action(str_action);
    if (action == 1){
      push_number(atoi(str_action));
    }
    else if (action == 2){
      operation_action(str_action[0]);
    }
    else if (action == 3){
      message_wrong_commande();
    }
  }
  quit_action();
  return 0;
}
