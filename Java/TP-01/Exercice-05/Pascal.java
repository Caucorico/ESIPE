public class Pascal
{

	public static int pascal(int nBut, int pBut)
	{
		int[] tab = new int[(nBut+1)];
		if ( tab == null )
		{
			System.err.println("Pas assez de place");
		}

		tab[0] = 1;

		for(int n = 1 ; n <= nBut ; n++ )
		{
			tab[n] = 1;

			for(int i = n-1 ; i>0 ; i--)
				tab[i] = tab[i-1] + tab[i];

		}

		int result = tab[pBut];
		return result;
	}

	/* En executant cette fonction, on remarque que le programme en Java est plus rapide qu'en C */
	/* Ce programme est plus rapide qu'en C car ... */
	public static void main(String[] args)
	{
		System.out.println(" Cn, p = " + Pascal.pascal(30000, 250));
	}
} 