import java.util.ArrayList;
import java.util.Arrays;

public class SynchronizedCOWList<E> {

    private E[] tab;

    private static final Object[] EMPTY = new Object[0];

    private final Object lock = new Object();

    @SuppressWarnings("unchecked")
    public SynchronizedCOWList() {
        synchronized (lock) {
            tab = (E[])EMPTY;
        }
    }

    public void add(E e) {
        synchronized (lock) {
            tab = Arrays.copyOf(tab, tab.length + 1);
            tab[tab.length - 1] = e;
        }
    }

    public int size() {
        synchronized (lock) {
            return tab.length;
        }
    }

    public static void main(String[] args) throws InterruptedException {
        var threads = new ArrayList<Thread>();
        var cowList = new SynchronizedCOWList<Integer>();

        for ( int i = 0 ; i < 4 ; i++ ) {
            var thread = new Thread(() -> {
                for ( int j = 0 ; j < 2_500 ; j++ ) {
                    cowList.add(j);
                }
            });
            thread.start();
            threads.add(thread);
        }

        for ( var thread : threads ) {
            thread.join();
        }

        System.out.println("Size = " + cowList.size());
    }
}
