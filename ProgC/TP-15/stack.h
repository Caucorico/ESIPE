#ifndef _STACK_
#define _STACK_

typedef struct _stack_element
{
  int element;
  struct _stack_element* next;
}stack_element;

typedef struct _stack
{
  stack_element* top;
  int size;
}stack;

stack_element* create_stack_element( int element );

void free_stack_element( stack_element* se );

stack* create_stack( void );

void free_stack( stack* s );

int push_stack_element( stack* s, stack_element* se );

int push_element( stack* s, int element );

stack_element* pop_stack_element( stack* s );

int display_stack( stack* s );

#endif 