package exercice01;

/* The default code already works : */
public class Person2 {
    private String name;
    private volatile int age;

    public Person2(String name, int age) {
        this.name = name;
        this.age = age;
    }
}
