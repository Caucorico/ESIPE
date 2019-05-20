package fr.umlv.fruits;

public class Pear implements Fruit
{
    /**
     * The juice factor of the pear.
     */
    private int juiceFactor;

    public Pear(int juiceFactor)
    {
        if ( juiceFactor < 0  || juiceFactor > 9 )
        {
            throw new IllegalArgumentException("The juice factor need to be between 0 and 9");
        }
        this.juiceFactor = juiceFactor;
    }


    @Override
    public int getPrice()
    {
        return 3*this.juiceFactor;
    }

    @Override
    public String toString()
    {
        return "Pear " + this.juiceFactor + "jf";
    }
}
