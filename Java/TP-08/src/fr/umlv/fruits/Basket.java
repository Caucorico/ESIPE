package fr.umlv.fruits;

import java.util.ArrayList;
import java.util.Objects;

public class Basket
{
    /**
     * The set of apple of the basket
     */
   private  ArrayList<FruitQuantity> fruits;

   public Basket()
   {
       this.fruits = new ArrayList<>();
   }

    public void add(Fruit fruit)
    {
        this.fruits.add(new FruitQuantity(Objects.requireNonNull(fruit), 1));
    }

   public void add(Fruit fruit, int quantity)
   {
       if ( quantity < 0 ) throw new IllegalArgumentException("The quantity need to be positive");

       this.fruits.add(new FruitQuantity(Objects.requireNonNull(fruit), quantity));
   }

    @Override
    public String toString()
    {
        int priceTotal = 0;
        StringBuilder sb = new StringBuilder();

        for ( FruitQuantity fruitq : this.fruits )
        {
            priceTotal += fruitq.getPrice();
            sb.append(fruitq.getFruit().toString());
            sb.append(" x ");
            sb.append(fruitq.getQuantity());
            sb.append('\n');
        }

        sb.append("price: ");
        sb.append(priceTotal/100);
        sb.append('\n');

        return sb.toString();
    }
}
