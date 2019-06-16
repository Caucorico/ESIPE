#ifndef _POL_CALC_
#define _POL_CALC_

#include "stack.h"

typedef enum _op
{
  OP_NUMBER, OP_OPERAND, OP_ACTION, OP_UNKNOWN
}op;

void listen_stdin( stack* s );

void execute_stack( void );

#endif