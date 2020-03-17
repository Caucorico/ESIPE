package exercice01;

public class Foo {
    private String value;

    public Foo(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }

    public static void main(String[] args) throws InterruptedException {
        Foo foo = new Foo("coucou");

        Thread t = new Thread(() -> {
            /* Some chances that value is null. */
            System.out.println(foo.getValue());
        });

        t.start();



    }
}
