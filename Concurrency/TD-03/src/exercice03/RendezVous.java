package exercice03;

import java.util.Objects;

public class RendezVous <V> {
    private V value;
    private final Object lock = new Object();

    private V getValue() {
        synchronized (lock) {
            return value;
        }
    }

    public void set(V value) {
        Objects.requireNonNull(value);
        this.value = value;
    }

    public V get() throws InterruptedException {
        while(getValue() == null);
        return value;
    }

    public static void main(String[] args) throws InterruptedException {
        RendezVous<String> rendezVous = new RendezVous<>();
        new Thread(() -> {
            try {
                Thread.sleep(5000);
            } catch (InterruptedException e) {
                throw new AssertionError(e);
            }
            rendezVous.set("hello");
        }).start();

        System.out.println(rendezVous.get());
    }
}
