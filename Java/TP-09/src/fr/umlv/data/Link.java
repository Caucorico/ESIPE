package fr.umlv.data;

/* Cette classe doit etre package private pour éviter que l'utilisateur modifie lui meme l'objet. */
/* Les champs de cette classe doivente etre private pour ne pas etre accessible depuis l'exterieur */
class Link
{
    /**
     * The value of the link
     */
    private int value;

    /**
     * Reference to the next element of the list
     */
    private Link next;

    Link( int value )
    {
        this.value = value;
        this.next = null;
    }

    Link( int value, Link next )
    {
        this.value = value;
        this.next = next;
    }

    @Override
    public String toString()
    {
        String s = "";
        s += ""+this.value;
        return s;
    }

    boolean hasNext()
    {
        return this.next != null;
    }

    Link next()
    {
        return this.next;
    }

    /* Pour éxécuter ce main, il suffit d'executer la commande suivante : java --class-path classes/ fr.umlv.data.Link */
    public static void main(String[] args)
    {
        Link l1 = new Link(13);
        Link l2 = new Link(144);
    }
}
