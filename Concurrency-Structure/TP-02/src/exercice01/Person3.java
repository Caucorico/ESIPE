package exercice01;
/*
Default code :

public class Person {
    private final String name;
    private final int age;

    public Person(String name, int age) {
      this.name = name;
      this.age = age;
      new Thread(() -> {
        System.out.println(this.name + " " + this.age);
      }).start();
    }
    ...
  }
 */

/* Access to the final parameters inside of the constructor is ot working. */
public class Person3 {
    private final String name;
    private final int age;

    public Person3(String name, int age) {
        this.name = name;
        this.age = age;
        new Thread(() -> {
            System.out.println(this.name + " " + this.age);
        }).start();
    }
}