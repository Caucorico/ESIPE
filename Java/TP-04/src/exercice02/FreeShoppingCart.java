package exercice02;

import java.util.ArrayList;

public class FreeShoppingCart
{
    /* Est-il intéressant de stocker le nombre maximum de livres dans un champ statique ?
     *
     * Non, ce n'est pas interessant car chaque Caddie peut avoir un nombre maximal de livre différent.
     * Le nombre de max dépend donc de l'objet et non pas de la classe. Il ne faut donc pas le mettre en statique.
     */

    /**
     * The max number of book in the ShoppingCart
     */
    private int maxBookNumber;

    /**
     * The book array of the ShoppingCart
     */
    private ArrayList<Book> booksArray;

    /**
     * FreeShoppingCart Constructor
     */
    public FreeShoppingCart()
    {
        this.booksArray = new ArrayList<Book>();
        this.maxBookNumber = 10;
    }

    /**
     * FreeShoppingCart Constructor
     * @param maxBookNumber The number max of books in the cart
     */
    public FreeShoppingCart(int maxBookNumber)
    {
        this.booksArray = new ArrayList<Book>();
        this.maxBookNumber = maxBookNumber;
    }


    /**
     * @param book The book to add to the list
     * @return return 0 when the book is successfully added and -1 when not
     */
    public int add(Book book)
    {
        if ( this.booksArray.size() < maxBookNumber )
        {
            this.booksArray.add(book);
            return 0;
        }
        else
        {
            return -1;
        }
    }

    /**
     * @return Return the number of books in the Cart
     */
    public int numberOfBooks()
    {
        return this.booksArray.size();
    }

    /**
     * @return Return the number of books and all the title and authors
     */
    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        sb.append("Number of books : ");
        sb.append(this.numberOfBooks());
        sb.append('\n');
        for ( Book b : this.booksArray )
        {
            sb.append(b.toString());
            sb.append('\n');
        }
        return sb.toString();
    }

    /**
     * @return Return the longest title in the Cart
     * @throws IllegalStateException Throw IllegalStateException when the list is empty
     */
    public String longestTitle() throws IllegalStateException
    {
        /* null for avoid eclipse code inspection :3  */
        String lt = null;
        int max = 0;

        if ( this.booksArray.isEmpty() ) throw new IllegalStateException("The list is empty!");

        for ( Book b : this.booksArray )
        {
            if ( b.getTitle().length() > max )
            {
                lt = b.getTitle();
                max = b.getTitle().length();
            }
        }

        return lt;
    }

    /**
     * @param toRemove The book to be remove of the list
     */
    public void removeAllOccurrences(Book toRemove )
    {
        for ( Book b : this.booksArray )
        {
            if ( b.equals(toRemove) )
            {
                this.booksArray.remove(b);
            }
        }
    }
}
