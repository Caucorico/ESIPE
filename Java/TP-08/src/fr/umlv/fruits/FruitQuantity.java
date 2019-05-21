package fr.umlv.fruits;

/**
 * That class connot be public because the other programmer don't need to use that class, they may use the Fruit interface.
 */
class FruitQuantity implements Fruit
{
    /**
     * The quantity of the fruit.
     */
    private int quantity;

    /**
     * The fruit.
     */
    private Fruit fruit;

    FruitQuantity(Fruit fruit, int quantity)
    {
        this.fruit = fruit;
        this.quantity = quantity;
    }

    int getQuantity()
    {
        return quantity;
    }

    Fruit getFruit()
    {
        return fruit;
    }

    @Override
    public int getPrice()
    {
        return this.quantity * this.getFruit().getPrice();
    }
}
