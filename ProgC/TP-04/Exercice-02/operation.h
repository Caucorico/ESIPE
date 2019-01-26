#ifndef _OPERATION_
#define _OPERATION_

#include "stack.h"
#include "interface.h"

int analyse_action(char* str_action);
void push_number(int n);
void operation_action(char op);
void quit_action(void);
void addition(void);
void soustraction(void);
void product(void);
void quotient(void);
void expo(void);
void message_wrong_commande(void);

#endif
