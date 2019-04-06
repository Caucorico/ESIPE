import java.util.ArrayList;

public class Main
 {
 	public static void main(String[] args)
 	{
		Book b1 = new Book("Da Java Code", "Duke Brown");
		Book b2 = b1;
		Book b3 = new Book("Da Java Code", "Duke Brown");

		/* Le code ci-dessous affiche true puis false. En effet, b1 et b2 contienne la meme reference tandis b3 contient un autre objet */
		/*System.out.println(b1 == b2);
		System.out.println(b1 == b3);*/

		ArrayList list = new ArrayList();
		list.add(b1);
		System.out.println(list.indexOf(b2));
		System.out.println(list.indexOf(b3));
		/* Aucun probleme :/ */

		Book aBook = new Book("Da Java Code", "Duke Brown");
		Book anotherBook = new Book(null, null);
		ArrayList list2 = new ArrayList();
		list2.add(aBook);
		System.out.println(list2.indexOf(anotherBook));
		/*
		 * Lorsque la methode equals est appele par indexOf, elle essaye d'acceder a la methode equals d'une string qui est null.
		 * Java leve donc une NullPointerException 
		 */

		/*
		 * Rappeler pourquoi un code doit arrêter de fonctionner si celui-ci est mal utilisé par un développeur.
		 * Pour eviter d'etre surpris par des erreurs ou des effets de bords plus tard dans le programme. 
		 */

		/*
		 * Que doit-on faire pour corriger le problème ? 
		 * Nous devons empecher les attributs de Book d'etre a null. Ou verifier si les attributs sont a null dans equals.
		 * Il vaut mieux privilegier la premiere option dans cette situation.
		 */

		/*
		 * Rappeler quelle est la règle de bonne pratique concernant l'utilisation de null.
		 * Nous devons utiliser null au minimum, uniquement lorsque c'est necessaire, c'est a dire 
		 * si un attribut est optionnel. ( Si on ajoutait couverture au livre et qu'il n'en a pas... )
		 */

		/*
		 * A quoi sert la méthode java.util.Objects.requireNonNull (RTFM) ?
		 * La methode sert a renvoyer le parametre si il n'est pas null et leve une NullPointerException si le parametre est null.
         *
		 * Comment l'utiliser ici pour empêcher de construire un livre avec des champs null ? 
		 * (=> Voir Book.java )
		 */
 	}
 }