package fr.uge.poo.ducks;

import java.util.ServiceLoader;

public class DuckFarmBetter {

    public static void main(String[] args) {
        ServiceLoader<DuckFactory> loader = ServiceLoader.load(DuckFactory.class);
        for(DuckFactory duckFactory : loader) {
            Duck[] ducks = new Duck[3];
            ducks[0] = duckFactory.withName("fifi");
            ducks[1] = duckFactory.withName("riri");
            ducks[2] = duckFactory.withName("Loulou");

            for ( var duck : ducks ) {
                System.out.println(duck.quack());
            }
        }
    }

}
