#include <stdio.h>
#include <string.h>
#include <math.h>

#include "operation.h"

/* Returns an integer which code a user action :
   - 0 : quit the programm 
   - 1 : push an element on the stack
   - 2 : resolve an arithmetic operation
   - 3 : invalid action */
int analyse_action(char* str_action){
  if (strncmp(str_action, "q", 2) == 0){
    return 0;
  }
  if (strcmp(str_action, "-") == 0){
    return 2;
  }
  if (((str_action[0] >= '1') && (str_action[0] <= '9')) || (str_action[0] == '-')){
    return 1;
  }
  if (strlen(str_action) == 1){
    switch (str_action[0]){
    case '+' :
    case '*' :
    case '/' :
    case '^' :
    case '%' :
    case '!' : return 2;
    default : return 3;
    }
  }
  return 3;
}

/* Animation when pushing a number on the stack. */
void push_number(int n){
  char msg[200];

  sprintf(msg, "Push in stack : %d", n);
  clear_window();
  display_action_msg(msg);
  stack_push(n);
  display_stack();
  actualise_window();
  pause_keyboard();
}

/* This fonction make the switch between the different operations. */
void operation_action(char op){
    switch (op){
    case '+' : addition(); break;
    case '-' : soustraction(); break;
    case '*' : product(); break;
    case '/' : quotient(); break;
    case '^' : expo(); break;
    case '%' :
    case '!' : return ;
    default : return ;
    }
}

/* To quit the programm. Display a message and pause for a second. */
void quit_action(void){
  clear_window();
  display_stack();
  display_action_msg("Good bye");
  actualise_window();
  pause_action();
  pause_action();
  free_windows();
}

/* Dealt with the addition. */
void addition(void){
  int a, b;

  clear_window();
  a = stack_pop();
  display_stack();
  display_transition_2(a);
  display_action_msg("Pop an element");
  actualise_window();
  pause_keyboard();

  clear_window();
  b = stack_pop();
  display_transition_2(a);
  display_transition_1(b);
  display_stack();
  display_action_msg("Pop a second element");
  actualise_window();
  pause_keyboard();

  clear_window();
  display_transition_2(a);
  display_transition_1(b);
  display_operation('+');
  display_stack();
  display_action_msg("Compute the sum");
  actualise_window();
  pause_keyboard();

  clear_window();
  display_stack();
  display_transition_1(a+b);
  display_action_msg("Push the sum");
  actualise_window();
  pause_keyboard();

  stack_push(a+b);
  clear_window();
  display_stack();
  actualise_window();
  pause_keyboard();
}

/* Dealt with the soustraction. */
void soustraction(void){
  int a, b;

  clear_window();
  a = stack_pop();
  display_stack();
  display_transition_2(a);
  display_action_msg("Pop an element");
  actualise_window();
  pause_keyboard();

  clear_window();
  b = stack_pop();
  display_transition_2(a);
  display_transition_1(b);
  display_stack();
  display_action_msg("Pop a second element");
  actualise_window();
  pause_keyboard();

  clear_window();
  display_transition_2(a);
  display_transition_1(b);
  display_operation('-');
  display_stack();
  display_action_msg("Compute the difference");
  actualise_window();
  pause_keyboard();

  clear_window();
  display_stack();
  display_transition_1(b-a);
  display_action_msg("Push the difference");
  actualise_window();
  pause_keyboard();

  stack_push(b-a);
  clear_window();
  display_stack();
  actualise_window();
  pause_keyboard();
}

/* Dealt with the product. */
void product(void){
  int a, b;

  clear_window();
  a = stack_pop();
  display_stack();
  display_transition_2(a);
  display_action_msg("Pop an element");
  actualise_window();
  pause_keyboard();

  clear_window();
  b = stack_pop();
  display_transition_2(a);
  display_transition_1(b);
  display_stack();
  display_action_msg("Pop a second element");
  actualise_window();
  pause_keyboard();

  clear_window();
  display_transition_2(a);
  display_transition_1(b);
  display_operation('x');
  display_stack();
  display_action_msg("Compute the product");
  actualise_window();
  pause_keyboard();

  clear_window();
  display_stack();
  display_transition_1(b*a);
  display_action_msg("Push the product");
  actualise_window();
  pause_keyboard();

  stack_push(b*a);
  clear_window();
  display_stack();
  actualise_window();
  pause_keyboard();
}

/* Dealt with the quotient. */
void quotient(void){
  int a, b;

  clear_window();
  a = stack_pop();
  display_stack();
  display_transition_2(a);
  display_action_msg("Pop an element");
  actualise_window();
  pause_keyboard();

  clear_window();
  b = stack_pop();
  display_transition_2(a);
  display_transition_1(b);
  display_stack();
  display_action_msg("Pop a second element");
  actualise_window();
  pause_keyboard();

  clear_window();
  display_transition_2(a);
  display_transition_1(b);
  display_operation('/');
  display_stack();
  display_action_msg("Compute the quotient");
  actualise_window();
  pause_keyboard();

  clear_window();
  display_stack();
  display_transition_1(b/a);
  display_action_msg("Push the quotient");
  actualise_window();
  pause_keyboard();

  stack_push(b/a);
  clear_window();
  display_stack();
  actualise_window();
  pause_keyboard();
}

/* Dealt with the exponentiation. */
void expo(void){
  int a, b;

  clear_window();
  a = stack_pop();
  display_stack();
  display_transition_2(a);
  display_action_msg("Pop an element");
  actualise_window();
  pause_keyboard();

  clear_window();
  b = stack_pop();
  display_transition_2(a);
  display_transition_1(b);
  display_stack();
  display_action_msg("Pop a second element");
  actualise_window();
  pause_keyboard();

  clear_window();
  display_transition_2(a);
  display_transition_1(b);
  display_operation('^');
  display_stack();
  display_action_msg("Compute the power");
  actualise_window();
  pause_keyboard();

  clear_window();
  display_stack();
  display_transition_1((int)pow(b,a));
  display_action_msg("Push the power");
  actualise_window();
  pause_keyboard();

  stack_push((int)pow(b,a));
  clear_window();
  display_stack();
  actualise_window();
  pause_keyboard();
}

/* Display the Wrong message comment on the screen. */
void message_wrong_commande(void){
  clear_window();
  display_stack();
  display_action_msg("Wrong command");
  actualise_window();
}
