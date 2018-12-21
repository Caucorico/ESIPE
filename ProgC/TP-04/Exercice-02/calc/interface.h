#ifndef _INTERFACE_
#define _INTERFACE_

#include "stack.h"

#define HEIGHT_WINDOWS_PIX 600
#define WIDTH_WINDOWS_PIX 800

#define WAIT_TIME_MILLISEC 500

void create_windows(void);
void free_windows(void);
void actualise_window(void);
void clear_window(void);
void pause_action(void);
void pause_keyboard(void);
void display_stack(void);
void display_transition_1(int number);
void display_transition_2(int number);
void display_operation(char op);
void display_action_msg(const char* msg);
void get_user_action(char* str_action);

#endif
