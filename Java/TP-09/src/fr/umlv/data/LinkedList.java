package fr.umlv.data;

/* L'interet d'utiliser un type paramétré ici est de sécurisé le type des objets qui seront stocké dans la liste. */
public class LinkedList <T>
{
    /**
     * Reference to the first element of the list.
     */
    private Link<T> first;

    /**
     * The size of the list.
     */
    private int size;

    public LinkedList()
    {
        this.first = null;
        this.size = 0;
    }

    public void add(T object)
    {
        this.first = new Link<>(object, this.first);
        this.size++;
    }

    @Override
    public String toString()
    {
        StringBuilder s;
        Link l = this.first;

        s = new StringBuilder("[");

        if ( this.first != null )
        {
            s.append(this.first.toString());
            while ( l.hasNext() )
            {
                s.append(",");
                l = l.next();
                s.append(l.toString());
            }
        }

        s.append("]");

        return s.toString();
    }

    public T get(int index)
    {
        if ( this.first == null )
        {
            throw new IllegalStateException("The list is empty");
        }
        else if ( index >= this.size )
        {
            throw new IllegalArgumentException("Index out of bound in the list");
        }
        else if ( index < 0 )
        {
            throw new IllegalArgumentException("The index cannot be negative");
        }
        Link<T> l = this.first;

        for ( int i = 0 ; i != index ; i++ )
        {
            l = l.next();
        }

        return l.getValue();
    }

    /* On utilise Object plutot que T ici car la méthode equals peut etre appelé sur les Objects? C'est une méthode de Object */
    public boolean contains(Object o)
    {
        if ( this.first == null ) return false;

        Link l = this.first;
        if ( o.equals(l.getValue())) return true;
        while ( l.hasNext() )
        {
            l = l.next();
            if ( o.equals(l.getValue())) return true;
        }

        return false;
    }
}
