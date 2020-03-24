package exercice01;

/* The default class already working. */
public class Person4 {
    private final String name;
    private final int age;

    public Person4(String name, int age) {
        this.name = name;
        this.age = age;
        new Thread(() -> {
            System.out.println(name + " " + age);
        }).start();
    }
}
