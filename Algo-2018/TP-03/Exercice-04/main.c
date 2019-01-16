#include <stdio.h>
#define SIZE 5

int horizontal_attack( int* tab, int pos)
{
	int i;
	for ( i = 0 ; i < pos ; i++ )
	{
		if ( tab[i] == tab[pos] )
		{
			return 1;
		}
	}
	return 0;
}

int diagonal_attack( int* tab, int size, int pos )
{
	int i;
	for ( i = 0 ; i < size ; i++ )
	{
		if ( pos-i == tab[pos]-tab[i] ||
			pos-i == tab[i]-tab[pos] )
		{
			return 1;
		}
	}
	return 0;
}

int conflict( int* tab, int size, int pos )
{
	int i;
	for ( i = 0 ; i < pos ; i++ )
	{
		if ( horizontal_attack(tab, pos) || diagonal_attack(tab, size, pos) )
		{
			return 1;
		}
	}

	return 0;
}

int nbr_queen( int* tab, int size, int pos )
{
	int i, count=0;

	if ( pos == size )
	{
		return 1;
	}

	for ( i = 0 ; i < size ; i++ )
	{
		tab[pos] = i;
		if ( !conflict(tab, size, pos) )
		{
			count = count+nbr_queen(tab, size, pos+1);
		}
	}

	return count;

}

int main( void )
{
	int tab[SIZE];
	int i, j;

	printf("%d \n",nbr_queen(tab, SIZE, 0));

	return 1;

}
