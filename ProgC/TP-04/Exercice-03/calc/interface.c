#include <MLV/MLV_all.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "interface.h"

void create_windows(void){
  MLV_create_window("Desk Calculator", NULL, WIDTH_WINDOWS_PIX, HEIGHT_WINDOWS_PIX);
}

void free_windows(void){
  MLV_free_window();
}

void actualise_window(void){
  MLV_actualise_window();
}

void clear_window(void){
  MLV_clear_window(MLV_rgba(0,0,0,255));
}

void pause_action(void){
  MLV_wait_milliseconds(WAIT_TIME_MILLISEC);
}

void pause_keyboard(void){
  MLV_wait_keyboard(NULL, NULL, NULL);
}

void display_stack(void){
  int i;
  int nb_elem = stack_size();
  char number[50];
  int length_char;

  /* Draw the border of the stack open at the top. */
  MLV_draw_line(20,20,20,20+(16*30), MLV_rgba(255,255,255,255));
  MLV_draw_line(20+80,20,20+80,20+(16*30), MLV_rgba(255,255,255,255));
  MLV_draw_line(20,20+(16*30),20+80,20+(16*30), MLV_rgba(255,255,255,255));

  /* Draw the element in the stack. */
  for (i=0 ; i<nb_elem ; i++){
    MLV_draw_line(20,20+((15-i)*30),20+80,20+((15-i)*30), MLV_rgba(255,255,255,255));
    MLV_draw_filled_rectangle(23, 23+((15-i)*30), 75, 25, MLV_rgba(255,55,55,255));
    length_char = sprintf(number, "%d", stack_get_element(i));
    MLV_draw_text(61-(4*length_char), 28+((15-i)*30), number, MLV_rgba(255,255,255,255));
  }
}

void display_action_msg(const char* msg){
  int length=strlen(msg);

  MLV_draw_text(450-(4*length), 150, msg, MLV_rgba(255,255,255,255));  
}

void display_transition_1(int number){
  char msg[50];
  int length_char;

  MLV_draw_filled_rectangle(250, 250, 75, 25, MLV_rgba(255,55,55,255));
  length_char = sprintf(msg, "%d", number);
  MLV_draw_text(288-(4*length_char), 255, msg, MLV_rgba(255,255,255,255));
}

void display_transition_2(int number){
  char msg[50];
  int length_char;

  MLV_draw_filled_rectangle(650, 250, 75, 25, MLV_rgba(255,55,55,255));
  length_char = sprintf(msg, "%d", number);
  MLV_draw_text(688-(4*length_char), 255, msg, MLV_rgba(255,255,255,255));
}

void display_operation(char op){
  char msg[50];

  sprintf(msg, "%c", op);
  MLV_draw_text(483, 255, msg, MLV_rgba(255,255,255,255));  
}

void get_user_action(char* str_action){
  char* str_action_adr = NULL;

  MLV_wait_input_box(300, 300, 300, 100,
		     MLV_rgba(255,255,255,255),
		     MLV_rgba(255,255,255,255),
		     MLV_rgba(255,55,55,255),
		     "Next action : ",
		     &str_action_adr);

  strncpy(str_action, str_action_adr, 255);
  str_action[255]='\0';
  free(str_action_adr);
}
