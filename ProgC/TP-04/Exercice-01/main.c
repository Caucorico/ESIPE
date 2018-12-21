#include "stack.h"
#include <stdio.h>

int main( void )
{
	int i, val;

	stack_init();
	stack_display();

	for ( i = 0 ; i < 30 ; i++)
	{
		stack_push(i*5);
		stack_display();
		putchar('\n');
	}

	for ( i = 29 ; i >= 0 ; i--)
	{
		val = stack_pop();
		printf("pop : %d \n", val);
		stack_display();
		putchar('\n');
	}


	return 0;
}