package exercice02;

import java.util.ArrayList;

public class UnboundedSafeQueue<V> {
    public final static int THREAD_NUMBER = 100;
    private final ArrayList<V> queue = new ArrayList<>();
    private final Object lock = new Object();
    private final int maxSize;

    public UnboundedSafeQueue(int queueSize) {
        this.maxSize = queueSize;
    }

    /**
     * @deprecated See put.
     * @param value The value to add at the last position of the queue.
     */
    public void add(V value) {
        synchronized (lock) {
            queue.add(value);
            lock.notifyAll();
        }
    }

    public void put(V value) throws InterruptedException {
        synchronized (lock) {
            while ( queue.size() >= maxSize ) {
                lock.wait();
            }
            queue.add(value);
            lock.notifyAll();
        }
    }

    public V take() throws InterruptedException {
        synchronized (lock) {
            while ( queue.size() < 1 ) {
                lock.wait();
            }
            lock.notifyAll();
            return queue.remove(0);
        }
    }

    public static void main(String[] args) {
        var queue = new UnboundedSafeQueue<String>(2);
        for( int i = 0 ; i < THREAD_NUMBER ; i++ ) {
            var thread = new Thread( () -> {
                while ( true ) {
                    try {
                        Thread.sleep(2_000);
                        queue.put(Thread.currentThread().getName());
                    } catch (InterruptedException e) {
                        throw new AssertionError(e);
                    }
                }
            });
            thread.setName("Thread " + (i+1));
            thread.start();
        }

        while ( true ) {
            try {
                System.out.println(queue.take());
            } catch (InterruptedException e) {
                throw new AssertionError(e);
            }
        }
    }
}
