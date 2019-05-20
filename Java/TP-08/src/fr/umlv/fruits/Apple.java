package fr.umlv.fruits;

import java.util.Objects;

public class Apple implements Fruit
{
    /**
     * The type of an apple.
     */
    private AppleKind type;

    /**
     * The apple mass in gram.
     */
    private int mass;

    public Apple(int mass, AppleKind type)
    {
        this.type = Objects.requireNonNull(type);

        if ( mass <= 0 )
        {
            throw new IllegalArgumentException("The mass need to greater than 0");
        }
        this.mass = mass;
    }

    @Override
    public String toString()
    {
        return this.type + " " + this.mass + " g";
    }

    public int getPrice()
    {
        return (mass*100)/2;
    }

    @Override
    public int hashCode()
    {
        return this.toString().hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        if ( !(obj instanceof Apple) ) return false;
        return this.type.equals(((Apple) obj).type) && this.mass == ((Apple) obj).mass;
    }
}
