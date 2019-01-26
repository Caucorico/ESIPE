#include "stack.h"
#include <stdio.h>

int main( void )
{
	int i, val;

	printf("########################################\n");
  printf("TP-04 Exercice-01. \nBut : Implementer les fonctions qui manipule une pile. \n\n");


  /* Zone TP */
	stack_init();
	stack_display();

	for ( i = 0 ; i < 20 ; i++)
	{
		printf("push %d :\n", i*5);
		stack_push(i*5);
		stack_display();
		putchar('\n');
	}

	for ( i = 20 ; i >= 0 ; i--)
	{
		val = stack_pop();
		printf("pop : %d \n", val);
		stack_display();
		putchar('\n');
	}

	/* Fin zone TP */


	printf("\n\n########################################\n");
	return 0;
}