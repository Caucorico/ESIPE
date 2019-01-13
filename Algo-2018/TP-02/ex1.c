#include <string.h>

int palindrome_rec(char str[], int lo, int hi)
{
	if ( lo >= hi )
	{
		return 1;
	}
	else
	{
		return ( str[lo] == str[hi] ) && palindrome_rec(str, lo+1, hi-1);
	}
}

int palindrome(char str[])
{
	return palindrome_rec(str, 0, strlen(str)-1 );
}