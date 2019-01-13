#include "ex4.h"

int digit_sum_digits_iter(int n)
{
	return sum_digits_iter(sum_digits_iter(n));
}

int digit_sum_digits_rec(int n)
{
	int res;
	if ( (n/10) == 0 ) return n%10;
	res = sum_digits_rec(n/10)+(n%10);
	res = digit_sum_digits_rec(res);
	return res;
}