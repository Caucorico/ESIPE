import java.util.Scanner; 

public class Calc { 
	public static void main(String[] args) {
		Scanner scanner = new Scanner(System.in); /* Variable scanner de Classe/type Scanner */
		int value = scanner.nextInt(); /* Variable value de type int */
		int value2 = scanner.nextInt();
		System.out.println("Entier 1 : "+value);
		System.out.println("Entier 2 : "+value2);
		System.out.println("Somme : "+(value+value2));
		System.out.println("Difference : "+(value-value2));
		System.out.println("Produit : "+(value*value2));
		System.out.println("Quotient : "+(value/value2));
		System.out.println("Reste : "+(value%value2));
	}
	/* nextInt n'est pas une fonction, c'est une methode de la classe Scanner. Ce n'est mas une fonction car elle est liée et dépend de l'objet Scanner */

	/* import java.util.Scanner; importe la classe Scanner pour éviter que nous ne devions recuperer le ficher et le placer dans nos sources */


}