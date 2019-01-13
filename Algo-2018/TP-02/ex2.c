#include <stdio.h>

void increasing_sequence_rec(int n)
{
	if ( n == 0 ) return;
	increasing_sequence_rec(n-1);
	printf("%d ", n);
}

void decreasing_sequence_rec(int n)
{
	if ( n == 0 ) return;
	printf("%d ", n);
	decreasing_sequence_rec(n-1);
}