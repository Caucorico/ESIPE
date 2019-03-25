public class Test
{
	/* On peut utiliser ' ' a la place de " " pour representer un character a la place d'une chaine. */
	public static void main(String[] args)
	{
         String first = args[0];
         String second = args[1];
         String last = args[2];
         System.out.println(first + ' ' + second + ' ' + last);
    }
    /* En ByteCode, on remarque que a chaque plus, on fait un invoeDynamic, on creer un objet intermediaire
       Du moins ca change en fonction des versions Java...
    */
      
}