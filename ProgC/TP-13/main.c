#include <stdio.h>
#include "bitboard.h"

int main(void)
{
	unsigned long int test;
	int i;

	test = 0x0;

	for ( i = 0 ; i < 64 ; i++ )
	{
		set_positive_bit_ULI(&test, i);
		print_ULI(test);
		set_negative_bit_ULI(&test, i);
		scanf("%*c");
		putchar('\n');
	}


	return 0;
}