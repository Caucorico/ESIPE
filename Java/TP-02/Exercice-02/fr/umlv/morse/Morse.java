package fr.umlv.morse;

public class Morse
{

	/*
	  11: iload         4
      13: iload_3
      14: if_icmpge     45
      17: aload_2
      18: iload         4
      20: aaload
      21: astore        5
      23: aload_1
      24: aload         5
      26: invokedynamic #3,  0      <======= 
      31: astore_1
      32: aload_1
      33: invokedynamic #4,  0      <======= 
      38: astore_1
      39: iinc          4, 1
      42: goto          11 <============ On remarque que les invokedynamics vont etre effectuer a la chaine dans une boucle...

      Dans quel cas doit-on utiliser StringBuilder.append() plutôt que le + 		
	  Il est donc preferable de l'utiliser dans de cas de figure, pour eviter de faire des invokedynamics en boucle

	  Et pourquoi est-ce que le chargé de TD va me faire les gros yeux si j'écris un + dans un appel à la méthode append?
	  Gain perdu
	 */
	public static void displayStopString(String[] args)
	{
		String s = "";

		for ( String elem : args )
		{
			s += elem;
			s += " Stop. ";
		}

		System.out.println(s);
	}

	/* La classe StringBuilder sert a manipuler les chaines de caracteres. Notamment a ajouter et inserer des elements. Elle permet d'eviter de devoir reafecter dans la memoire une string a chaque modification
	 * La  methode append renvoie this. C'est pourquoi elle renvoie un StringBuilder. Permet d'enchainer les append sans etre obliger de devoir reecrire la reference vers l'objet a chaque fois 
	 */

	public static void main(String[] args)
	{
		Morse.displayStopString(args);
	}
}