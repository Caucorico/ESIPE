package fr.uge.poo.paint.rapport.ducks;

import java.util.ServiceLoader;

public class DuckFarm {

    public static void main(String[] args) {
        ServiceLoader<Duck> loader = ServiceLoader.load(Duck.class);
        for(Duck duck : loader) {
            System.out.println(duck.quack());
        }
    }

}
