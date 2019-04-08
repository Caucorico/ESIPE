package exercice02;

public class Main
{
    public static void main(String[] args)
    {
        FreeShoppingCart a = new FreeShoppingCart();
        Book harry = new Book("Harry Potter and the philosophal stone", "J. K. Rolling");
        Book b2 = new Book("Maths", "J. F. Mathematician");
        a.add(harry);
        a.add(b2);
        System.out.println(a.toString());

        System.out.println("Longest title : " + a.longestTitle());
    }
}
