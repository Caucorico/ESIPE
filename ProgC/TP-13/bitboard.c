#include <stdio.h>
#include "bitboard.h"

int bit_value_ULI(unsigned long int n, int position)
{
	unsigned long int mask;

	mask = 0x1;

	mask <<= position;
	mask &= n;
	mask >>= position;

	return mask;
}

void print_ULI(unsigned long int n)
{
	int i;

	for ( i = 63 ; i >= 0 ; i-- )
	{
		printf("%d ", bit_value_ULI(n, i));
		if ( i%8 == 0 ) putchar('\n');
	}
}

void set_positive_bit_ULI(unsigned long int* n, int position)
{
	unsigned long int mask;

	mask = 0x1;

	mask <<= (position);

	*n |= mask;
}

void set_negative_bit_ULI(unsigned long int* n, int position)
{
	unsigned long int mask;

	mask = 0x1;

	mask <<= position;

	mask = ~mask;

	*n &= mask;
}