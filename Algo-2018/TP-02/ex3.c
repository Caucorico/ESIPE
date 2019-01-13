int count(int t[], int lo, int hi, int elt)
{
	if ( lo > hi ) return 0;
	else if ( t[hi] == elt )
	{
		return 1 + count(t, lo, hi-1, elt );
	}
	else
	{
		return count(t, lo, hi-1, elt );
	}
}

int max_count(int t[], int lo, int hi)
{
	int current_nbr, child_max_nbr;
	if ( lo > hi ) return 0;

	current_nbr = count(t, lo, hi, t[lo]);

	child_max_nbr = max_count(t, lo+1, hi );

	if ( current_nbr < child_max_nbr )
	{
		return child_max_nbr;
	}
	else
	{
		return current_nbr;
	}
}