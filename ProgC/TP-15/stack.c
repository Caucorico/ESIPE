#include <stdio.h>
#include <stdlib.h>
#include "stack.h"

stack_element* create_stack_element( int element )
{
  stack_element* new_stack_element;

  new_stack_element = malloc(sizeof(stack_element));

  if ( new_stack_element == NULL )
  {
    perror("Error in stack.c in create_stack_element ");
    return NULL;
  }

  new_stack_element->element = element;

  new_stack_element->next = NULL;

  return new_stack_element;
}

void free_stack_element( stack_element* se )
{
  if ( se != NULL )
  {
    free( se );
  }
}

stack* create_stack( void )
{
  stack* new_stack;

  new_stack = malloc(sizeof(stack));

  if ( new_stack == NULL )
  {
    perror("Error in stack.c in create_stack ");
    return NULL;
  }

  new_stack->top = NULL;
  new_stack->size = 0;

  return new_stack;
}

void free_stack( stack* s )
{
  stack_element* buff;
  stack_element* buff2;

  if ( s != NULL )
  {
    buff = s->top;
    while ( buff != NULL )
    {
      buff2 = buff;
      buff = buff->next;
      free(buff2);
    }
    free(s);
  }
}

int push_stack_element( stack* s, stack_element* se )
{
  if ( s == NULL )
  {
    fprintf(stderr, "Error in stack.c in push_stack_element : s is NULL\n");
    return -1;
  }
  else if ( se == NULL )
  {
    fprintf(stderr, "Error in stack.c in push_stack_element : se is NULL\n");
    return -2;
  }

  se->next = s->top;
  s->top = se;
  s->size++;

  return 0;
}

int push_element( stack* s, int element )
{
  stack_element* new_stack_element;
  int err;

  if ( s == NULL )
  {
    fprintf(stderr, "Error in stack.c in push_element : s is NULL\n");
    return -1;
  }

  new_stack_element = create_stack_element(element);
  if ( new_stack_element == NULL )
  {
    fprintf(stderr, "Error in stack.c in push_element : new_stack_element was not created\n");
    return -3;
  }

  err = push_stack_element( s, new_stack_element );
  if ( err < 0 )
  {
    fprintf(stderr, "Error in stack.c in push_element : push_stack_element has failed\n");
    return -4;
  }

  return 0;

}

stack_element* pop_stack_element( stack* s )
{
  stack_element* buff;

  if ( s == NULL )
  {
    fprintf(stderr, "Error in stack.c in pop_stack_element : s is NULL\n");
    return NULL;
  }
  else if ( s->size <= 0 )
  {
    fprintf(stderr, "Error in stack.c in pop_stack_element : The stack is empty !\n");
    return NULL;
  }

  buff = s->top;
  s->top = s->top->next;
  s->size--;
  return buff;
}

void display_stack_elements( stack_element* se )
{
  if ( se == NULL ) return;
  display_stack_elements(se->next);
  printf("%d ", se->element );

}

int display_stack( stack* s )
{
  if ( s == NULL )
  {
    fprintf(stderr, "Error in stack.c in display_stack : s is NULL !\n");
    return -1;
  }

  display_stack_elements(s->top);
  putchar('\n');

  return 0;
} 