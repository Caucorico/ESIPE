package exercice01;

/*
Default code :

public class Person {
    private String name;
    private final int age;

    public Person(String name, int age) {
      this.name = name;
      this.age = age;
    }
    ...
  }
 */

/* Corrected class : */
public class Person1 {
    private final String name;
    private final int age;

    public Person1(String name, int age) {
        this.name = name;
        this.age = age;
    }
}
