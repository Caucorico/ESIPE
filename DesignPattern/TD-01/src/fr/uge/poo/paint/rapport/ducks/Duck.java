package fr.uge.poo.paint.rapport.ducks;

public interface Duck {

    String quack();

    Duck setName(String name);

    Duck clone();

}
