package fr.umlv.fruits.main;

import fr.umlv.fruits.Apple;
import fr.umlv.fruits.AppleKind;
import fr.umlv.fruits.Basket;
import fr.umlv.fruits.Pear;

import java.util.HashSet;

public class Main
{
    public static void main(String[] args) {
        /*Apple apple1 = new Apple(20, "Golden");
        Apple apple2 = new Apple(40, "Pink Lady");

        Basket basket = new Basket();
        basket.add(apple1);
        basket.add(apple2);
        System.out.println(basket);*/

        /*HashSet<Apple> set = new HashSet<>();
        set.add(new Apple(20, "Golden"));
        System.out.println(set.contains(new Apple(20, "Golden")));*/

        /*Apple apple1 = new Apple(20, "Golden");
        Apple apple2 = new Apple(40, "Pink Lady");
        Pear pear = new Pear(5);

        Basket basket = new Basket();
        basket.add(apple1);
        basket.add(apple2);  // une pomme
        basket.add(pear);    // une poire
        System.out.println(basket);*/

        /*Apple apple1 = new Apple(20, "Golden");
        Apple apple2 = new Apple(40, "Pink Lady");
        Pear pear = new Pear(5);

        Basket basket = new Basket();
        basket.add(apple1, 5);      // 5 pommes
        basket.add(apple2);
        basket.add(pear, 7);        // 7 poires
        System.out.println(basket);*/

        Apple apple1 = new Apple(20, AppleKind.Golden);
        Apple apple2 = new Apple(40, AppleKind.Pink_Lady);
        Pear pear = new Pear(5);

        Basket basket = new Basket();
        basket.add(apple1, 5);
        basket.add(apple2);
        basket.add(pear, 7);
        System.out.println(basket);
    }

}
