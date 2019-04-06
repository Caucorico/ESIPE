#include <stdio.h>

void display_tab(int tab[], int size)
{
	int i;
	printf("[");
	for ( i = 0 ; i < size-1 ; i++ )
	{
		printf(" %d,", tab[i]);
	}
	printf(" %d ]\n", tab[i]);
}

void permutations(int buffer[], int current, int max)
{
	int i;

	/*for ( i = 0 ; i < current ; i++ ) printf("    ");
	printf("-->");
	display_tab(buffer, max);*/

	if ( current > max )
	{
		display_tab(buffer, max);
		return;
	}

	for ( i = 0 ; i < max ; i++ )
	{
		if ( buffer[i] == 0 )
		{
			buffer[i] = current;
			permutations(buffer, current+1, max);
			buffer[i] = 0;
		}
	}
}

int main(void)
{
	int buffer[3] = { 0, 0, 0 };

	printf("########################################\n");
  printf("TP-06 Exercice-02. \nBut : Lister toutes les permutations possible d'au tableau d'entier. \n\n");

  /* Zone TP */

	permutations(buffer, 1, 3);

	/* Fin zone TP */
	printf("\n\n########################################\n");

	return 0;
}