package fr.uge.poo.paint.rapport.ducks;

import java.util.ServiceLoader;

public class DuckFarmBetter {

    public static void main(String[] args) {
        ServiceLoader<Duck> loader = ServiceLoader.load(Duck.class);
        for(Duck duck : loader) {
            Duck[] ducks = new Duck[3];
            ducks[0] = duck.setName("fifi");
            ducks[1] = duck.clone().setName("riri");
            ducks[2] = duck.clone().setName("loulou");

            for ( var d : ducks ) {
                System.out.println(d.quack());
            }
        }
    }

}
