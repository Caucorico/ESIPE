#include <stdio.h>
#define SIZE 9

void zeroing(int tab[SIZE][SIZE], int size)
{
	int i,j;
	for ( i = 0 ; i < size ; i++ )
	{
		for ( j = 0 ; j < size ; j++ )
		{
			tab[i][j] = 0;
		}
	}
}

void solving(int tab[SIZE][SIZE], int current, int max)
{
	if ( current > max )
	{
		/* do some code here */
	}

	for ( i = 0 ; i < max ; i++ )
	{
		for ( j = 0 ; j < max ; j++ )
		{
			if ( tab[i][j] == 0 )
			{
				tab[i][j] = current;
				solving(tab, current+1, max);
				tab[i][j] = 0;
			}
		}
	}
}

int main(void)
{
	int i,j;
	int test[SIZE][SIZE];
	zeroing(test, SIZE);



	return 0;
}