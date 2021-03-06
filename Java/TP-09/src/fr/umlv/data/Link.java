package fr.umlv.data;

/* Cette classe doit etre package private pour éviter que l'utilisateur modifie lui meme l'objet. */
/* Les champs de cette classe doivente etre private pour ne pas etre accessible depuis l'exterieur */
class Link <T>
{
    /**
     * The object of the link
     */
    private T object;

    /**
     * Reference to the next element of the list
     */
    private Link<T> next;

    Link( T object )
    {
        this.object = object;
        this.next = null;
    }

    Link( T object, Link<T> next )
    {
        this.object = object;
        this.next = next;
    }

    @Override
    public String toString()
    {
        String s = "";
        s += ""+this.object.toString();
        return s;
    }

    boolean hasNext()
    {
        return this.next != null;
    }

    Link<T> next()
    {
        return this.next;
    }

    T getValue()
    {
        return this.object;
    }

    /* Pour éxécuter ce main, il suffit d'executer la commande suivante : java --class-path classes/ fr.umlv.data.Link */
    public static void main(String[] args)
    {
        Link l1 = new Link<>(13);
        Link l2 = new Link<>(144);
    }
}
