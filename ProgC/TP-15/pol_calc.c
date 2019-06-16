#include <stdio.h>
#include <stdlib.h>
#include <readline/readline.h>
#include <readline/history.h>
#include "pol_calc.h"
#include "stack.h"

op get_operand(char* element)
{
  if ( strspn(element, "0123456789") == strlen(element) )
  {
    return OP_NUMBER;
  }
  else if ( strspn(element, "+-*/%^!") == 1 )
  {
    return OP_OPERAND;
  }
  else if ( strspn(element, "qpcar") == 1 )
  {
    return OP_ACTION;
  }
  else
  {
    return OP_UNKNOWN;
  }
}

int execute_operand(stack* s, char* element)
{
  int a,b,c,i;
  stack_element* se;

  if ( strcmp(element, "+") == 0 )
  {
    if ( s->size < 2 )
    {
      fprintf(stderr, "There are no enough arguments for \"+\" \n");
      return -1;
    }

    se = pop_stack_element(s);
    if ( se == NULL )
    {
      fprintf(stderr, "Error in pol_calc.c in execute_operand : error in pop\n" );
      return -2;
    }
    a = se->element;
    free_stack_element(se);

    se = pop_stack_element(s);
    if ( se == NULL )
    {
      fprintf(stderr, "Error in pol_calc.c in execute_operand : error in pop\n" );
      return -2;
    }
    b = se->element;
    free_stack_element(se);

    push_element(s, b+a);
  }
  else if ( strcmp(element, "-") == 0 )
  {
    if ( s->size < 2 )
    {
      fprintf(stderr, "There are no enough arguments for \"-\" \n");
      return -1;
    }

    se = pop_stack_element(s);
    if ( se == NULL )
    {
      fprintf(stderr, "Error in pol_calc.c in execute_operand : error in pop\n" );
      return -2;
    }
    a = se->element;
    free_stack_element(se);

    se = pop_stack_element(s);
    if ( se == NULL )
    {
      fprintf(stderr, "Error in pol_calc.c in execute_operand : error in pop\n" );
      return -2;
    }
    b = se->element;
    free_stack_element(se);

    push_element(s, b-a);
  }
  else if ( strcmp(element, "*") == 0 )
  {
    if ( s->size < 2 )
    {
      fprintf(stderr, "There are no enough arguments for \"*\" \n");
      return -1;
    }

    se = pop_stack_element(s);
    if ( se == NULL )
    {
      fprintf(stderr, "Error in pol_calc.c in execute_operand : error in pop\n" );
      return -2;
    }
    a = se->element;
    free_stack_element(se);

    se = pop_stack_element(s);
    if ( se == NULL )
    {
      fprintf(stderr, "Error in pol_calc.c in execute_operand : error in pop\n" );
      return -2;
    }
    b = se->element;
    free_stack_element(se);

    push_element(s, b*a);
  }
  else if ( strcmp(element, "/") == 0 )
  {
    if ( s->size < 2 )
    {
      fprintf(stderr, "There are no enough arguments for \"/\" \n");
      return -1;
    }

    se = pop_stack_element(s);
    if ( se == NULL )
    {
      fprintf(stderr, "Error in pol_calc.c in execute_operand : error in pop\n" );
      return -2;
    }
    a = se->element;
    free_stack_element(se);

    se = pop_stack_element(s);
    if ( se == NULL )
    {
      fprintf(stderr, "Error in pol_calc.c in execute_operand : error in pop\n" );
      return -2;
    }
    b = se->element;
    free_stack_element(se);

    push_element(s, b/a);
  }
  else if ( strcmp(element, "%") == 0 )
  {
    if ( s->size < 2 )
    {
      fprintf(stderr, "There are no enough arguments for \" %% \" \n");
      return -1;
    }

    se = pop_stack_element(s);
    if ( se == NULL )
    {
      fprintf(stderr, "Error in pol_calc.c in execute_operand : error in pop\n" );
      return -2;
    }
    a = se->element;
    free_stack_element(se);

    se = pop_stack_element(s);
    if ( se == NULL )
    {
      fprintf(stderr, "Error in pol_calc.c in execute_operand : error in pop\n" );
      return -2;
    }
    b = se->element;
    free_stack_element(se);

    push_element(s, b%a);
  }
  else if ( strcmp(element, "^") == 0 )
  {
    if ( s->size < 1 )
    {
      fprintf(stderr, "There are no enough arguments for \"^\" \n");
      return -1;
    }

    se = pop_stack_element(s);
    if ( se == NULL )
    {
      fprintf(stderr, "Error in pol_calc.c in execute_operand : error in pop\n" );
      return -2;
    }
    a = se->element;
    free_stack_element(se);

    se = pop_stack_element(s);
    if ( se == NULL )
    {
      fprintf(stderr, "Error in pol_calc.c in execute_operand : error in pop\n" );
      return -2;
    }
    b = se->element;
    free_stack_element(se);

    c = b;
    for ( i = 1 ; i < a ; i++ )
    {
      c *= b;
    }
    push_element(s,c);
  }
  else if ( strcmp(element, "!") == 0 )
  {
    if ( s->size < 1 )
    {
      fprintf(stderr, "There are no enough arguments for \"!\" \n");
      return -1;
    }

    se = pop_stack_element(s);
    if ( se == NULL )
    {
      fprintf(stderr, "Error in pol_calc.c in execute_operand : error in pop\n" );
      return -2;
    }
    a = se->element;
    free_stack_element(se);
    if ( a < 0 )
    {
      fprintf(stderr, "Error in pol_calc.c in execute_operand : factorial connot be negative\n" );
      return -3;
    }

    b = 1;
    for ( i = 0 ; i <= a ; i++ )
    {
      b *= (i+1);
    }

    push_element(s, b);
  }


  return 0;
}

int push_number_element(stack* s, const char* element)
{
  int number, err;

  number = atoi(element);

  err = push_element(s, number);

  if ( err < 0 )
  {
    return err;
  }

  return 0;
}

int execute_action(stack* s, const char* element)
{
  stack_element* buff1;
  stack_element* buff2;

  if ( element == NULL )
  {
    fprintf(stderr, "Error in pol_calc.c in execute_action : element connot be null\n" );
    return -1;
  }
  else if ( s == NULL )
  {
    fprintf(stderr, "Error in pol_calc.c in execute_action : s connot be null\n" );
    return -2;
  }

  if ( strcmp(element, "a") == 0 )
  {
    if ( s->size == 0 )
    {
      fprintf(stdout, "The stack is empty !\n");
    }
    else
    {
      display_stack( s );
    }
  }
  else if ( strcmp(element, "p") == 0 )
  {
    if ( s->size == 0 )
    {
      fprintf(stdout, "The stack is empty !\n");
    }
    else
    {
      fprintf(stdout, "%d\n", s->top->element );
    }
  }
  else if ( strcmp(element, "r") == 0 )
  {
    if ( s->size < 2 )
    {
      fprintf(stdout, "There are no enough argument to reverse !\n");
    }
    else
    {
      buff1 = pop_stack_element(s);
      buff2 = pop_stack_element(s);
      push_stack_element(s, buff1);
      push_stack_element(s, buff2);
    }
  }
  else if ( strcmp(element, "c") == 0 )
  {
    while ( s->size > 0 )
    {
      free_stack_element(pop_stack_element(s));
    }
  }
}

void listen_stdin( stack* s )
{
  char* line;
  int index = 0;
  char element[15];
  int err;
  op operand;

  while ( strcmp(element, "q") != 0 )
  {
    line = readline(NULL);

    do
    {
      err = sscanf(&line[index], "%s", element);
      if ( err > 0 )
      {
        index += (strlen(element)+1);
        
        operand = get_operand(element);

        if ( operand == OP_NUMBER )
        {
          push_number_element(s, element);
        }
        else if ( operand == OP_OPERAND )
        {
          execute_operand(s, element);
        }
        else if ( operand == OP_ACTION )
        {
          execute_action(s, element );
        }
        else
        {
          printf("This option isn't available : %s\n", element);
        }
      }
    }while ( strcmp(element, "q") != 0 && err > 0 && index < (int)strlen(line));
    index = 0;
    free(line);
  }
}

void execute_stack( void )
{
  stack* s;

  s = create_stack();

  listen_stdin(s);

  free_stack(s);
}