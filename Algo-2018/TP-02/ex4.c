int sum_digits_iter(int n)
{
	int number=0;

	while ( n/10 != 0 )
	{
		number += n%10;
		n /= 10;
	}
	number += n%10;
	return number;
}

int sum_digits_rec(int n)
{
	if ( (n/10) == 0 ) return n%10;
	return sum_digits_rec(n/10)+(n%10);
}