package exercice02;

public class ExampleLongAffectation {
    long l = -1L;

    public static void main(String[] args) {
        var e = new ExampleLongAffectation();
        new Thread(() -> {
            System.out.println("l = " + e.l);
        }).start();
        e.l = 0;
    }
}
