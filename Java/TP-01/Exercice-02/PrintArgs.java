public class PrintArgs
{
	public static void main(String[] args)
	{
		/* Lorsque aucun argument n'est passé au programme, le tableau args est vide. La ligne ci-dessous affiche donc l'erreur java.lang.ArrayIndexOutOfBoundsException */
		// System.out.println(args[0]);

		for ( int i = 0 ; i < args.length ; i++ )
		{
			System.out.println(args[i]);
		}

		for ( String elem : args )
		{
			System.out.println(elem);
		}
	}
}