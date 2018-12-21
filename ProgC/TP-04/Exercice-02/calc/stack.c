#include "stack.h"
#include <stdio.h>

static Stack stack;

void stack_init( void )
{
	stack.size = 0;
}

int stack_size(void)
{
	return stack.size;
}

int stack_is_empty(void)
{
	if ( stack_size() > 0 )
	{
		return 0;
	}
	else
	{
		return 1;
	}
}

int stack_top(void)
{
	return stack.values[stack.size-1];
}

int stack_pop(void)
{
	int element;

	if ( !stack_is_empty() )
	{
		element = stack.values[stack.size-1];
		stack.size--;
	}
	else
	{
		printf("Stack empty\n");
	}
	return element;
}

void stack_push(int n)
{
	if ( stack.size < MAX_SIZE )
	{
		stack.values[stack.size] = n;
		stack.size++;
	}
	else
	{
		printf("Stack full\n");
	}
}

void stack_display(void)
{
	int i;

	if ( stack.size == 0 )
	{
		printf("stack empty\n");
	}

	for (i = stack.size-1 ; i >= 0  ; i--)
	{
		printf("stack[%d] = %d \n", i, stack.values[i] );
	}
}

int stack_get_element(int index)
{
	return stack.values[index];
}