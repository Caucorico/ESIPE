package exercice02;

public class ExempleReordering {
    int a = 0;
    int b = 0;

    public static void main(String[] args) {
        var e = new ExempleReordering();
        new Thread(() -> { System.out.println("a = " + e.a + "  b = " + e.b); }).start();
        e.a = 1;
        e.b = 2;
    }
}
