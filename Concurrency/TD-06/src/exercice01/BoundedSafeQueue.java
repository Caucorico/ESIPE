package exercice01;

import java.util.ArrayDeque;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

public class BoundedSafeQueue<V> {

    public final static int THREAD_NUMBER = 100;
    private final ArrayDeque<V> fifo = new ArrayDeque<>();
    private final int capacity;

    private final ReentrantLock lock = new ReentrantLock(false);
    private final Condition waitAccessQueue = lock.newCondition();

    public BoundedSafeQueue(int capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException();
        }
        this.capacity = capacity;
    }

    public void put(V value) throws InterruptedException {
        lock.lock();
        try {
            while (fifo.size() == capacity) {
                waitAccessQueue.await();
            }
            fifo.add(value);
            waitAccessQueue.signalAll();
        } finally {
            lock.unlock();
        }
    }

    public V take() throws InterruptedException {
        lock.lock();
        try {
            while (fifo.isEmpty()) {
                waitAccessQueue.await();
            }
            waitAccessQueue.signalAll();
            return fifo.remove();
        } finally {
            lock.unlock();
        }
    }

    public static void main(String[] args) {
        var queue = new BoundedSafeQueue<String>(5);
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
