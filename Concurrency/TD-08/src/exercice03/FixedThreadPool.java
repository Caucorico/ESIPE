package exercice03;

import java.util.Arrays;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class FixedThreadPool {

    final private int poolsize;
    final private LinkedBlockingQueue<Task> tasksQueue;
    final private Thread[] threads;

    public FixedThreadPool(int poolsize) {
        this.poolsize = poolsize;
        this.threads = new Thread[poolsize];
        this.tasksQueue = new LinkedBlockingQueue<>();
    }

    public void start(){
        IntStream.range(0, poolsize)
            .forEach( e -> {
                Thread thread = null;
                thread = new Thread(() -> {
                    try {
                        tasksQueue.take().run();
                    } catch (InterruptedException ex) {
                        Thread.currentThread().interrupt();
                    }
                });
                thread.start();
                threads[e] = thread;
            });
    }

    public void submit(Task r){
        tasksQueue.offer(r);
    }

    public void stop(){
        Arrays.stream(threads).forEach(Thread::interrupt);
    }

    


}
