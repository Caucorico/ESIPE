#ifndef _CALC_ENTRY_
#define _CALC_ENTRY_

#include <stdio.h>
#include <string.h>
#include "calc.h"
#include "dispatch.h"

typedef enum operand
{
	ADDITION,
	SUBSTRACTION,
	MULTIPLICATION
}operand;

typedef struct operation
{
	operand op;
	int a;
	int b;
}operation;

/* This function return the */
int parse_line(char* line, operation* oper);

int call_function_by_operation(operation oper);

int listen_operation(void);

#endif
