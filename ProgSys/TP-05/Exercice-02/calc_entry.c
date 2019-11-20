#include "calc_entry.h"

int parse_line(char* line, operation* oper)
{
	char buff[10];
	sscanf(line, "%s %d %d", buff, &oper->a, &oper->b);
	if ( strcmp("add", buff) == 0 )
	{
		oper->op = ADDITION;
		return 1;
	}
	else if ( strcmp("sub", buff) == 0 )
	{
		oper->op = SUBSTRACTION;
		return 1;
	}
	else if ( strcmp("mult", buff) == 0 )
	{
		oper->op = MULTIPLICATION;
		return 1;
	}

	return 0;
}

int call_function_by_operation(operation oper)
{
	if ( oper.op == ADDITION )
	{
		startProc()
	}

}
