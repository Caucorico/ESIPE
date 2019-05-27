package fr.umlv.data;

public class LinkedList
{
    /**
     * Reference to the first element of the list.
     */
    private Link first;

    public LinkedList()
    {
        this.first = null;
    }

    public void add(int value)
    {
        this.first = new Link(value, this.first);
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
}
