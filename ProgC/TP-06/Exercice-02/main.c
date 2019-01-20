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

int main(int argc, char const *argv[])
{
	int buffer[3] = { 0, 0, 0 };
	permutations(buffer, 1, 3);
	return 0;
}