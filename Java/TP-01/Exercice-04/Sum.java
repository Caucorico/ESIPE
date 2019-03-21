import java.util.Arrays;

public class Sum
{
	/* Cette methode est statique car elle ne dépend pas de l'objet Sum (this n'appait pas)*/
	/* Attention, lorsque le tableau d'entier passer en argument contient une chaine qui n'est pas un nombre, java lève l'exception java.lang.NumberFormatException */
	public static int[] getIntArrayByStringArray(String[] stringArray)
	{
		int[] intArray = new int[stringArray.length];
		for ( int i = 0 ; i < stringArray.length ; i++ )	
		{
			intArray[i] = Integer.parseInt(stringArray[i]);
		}

		return intArray;
	}

	public static int getSumOfIntArray(int[] intArray)
	{
		int sum = 0;

		for ( int i : intArray )
		{
			sum += i;
		}

		return sum;
	}

	public static void main(String[] args)
	{
		int[] intArray = Sum.getIntArrayByStringArray(args);
		int sum = getSumOfIntArray( intArray );

		System.out.println("Tableau d'entier : " + Arrays.toString(intArray));
		System.out.println("Somme des nombres : " + sum);
	}
}