public class Exo1 {

	/* Le code suivant affiche true puis false */
	public static void question1()
	{
		String s1 = "toto";
		String s2 = s1;
		String s3 = new String(s1);

		System.out.println(s1 == s2); /* affiche true car s1 et s2 contienne la m^eme référence vers l'objet String qui est comparé ici*/
		System.out.println(s1 == s3); /* affiche false car s3 est un objet différent de s1 et s2 et n'a donc pas la m^eme réference */
	}

	public static void question2()
	{
		String s4 = "toto";
		String s5 = new String(s4);

		System.out.println( s4.equals(s5)); /* Il faut utiliser la méthode equals de la classe String */
	}

	/* Cette fonction affiche true. s6 et s7 contienne la m^eme reference vers la string "toto". En affectation directe, pour éviter les doublons l'objet String contient la reference a la chaine unique.  */
	public static void question3()
	{
		String s6 = "toto";
		String s7 = "toto";

		System.out.println(s6 == s7);
	}
	/* Les String ne doivent pas ^etre mutable pour eviter les effets de bords. Il serait embettant de modifier tous les utilisateur toto en voulant ne changer qu'un seul toto */

	/* Le code suivant affiche hello. En effet, l'objet String etant non mutable, la methode toUpperCase qui est cense la modifiee renvoie un nouvel objet String, s8 n'est donc pas modifie */
	public static void question5()
	{
		String s8 = "hello";
		s8.toUpperCase();
		System.out.println(s8);
	}

	public static void main(String[] args)
	{
		Exo1.question5();
	}
}