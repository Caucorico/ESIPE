package exercice02;

import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicInteger;

public class Counter {
    private final AtomicInteger counter = new AtomicInteger();

    public int nextInt() {
        int i;
        do { i = counter.get(); }
        while (!counter.compareAndSet(i, i+1) );
        return i;
    }

    public int nextInt2() {
        return counter.getAndIncrement();
    }

    public static void main(String[] args) throws InterruptedException {
        Counter counter = new Counter();
        ArrayList<Thread> threads = new ArrayList<>();

        for ( var i = 0 ; i < 4 ; i++ ) {
            var thread = new Thread(() -> {
               for ( var j = 0 ; j < 100_000 ; j++ ) {
                    counter.nextInt2();
                }
            });

            thread.start();
            threads.add(thread);
        }

        for ( var thread : threads ) {
            thread.join();
        }

        System.out.println(counter.counter);
    }
}

