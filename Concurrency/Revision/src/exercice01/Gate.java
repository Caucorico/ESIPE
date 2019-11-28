package exercice01;

import java.util.stream.IntStream;

public class Gate {

    private final int nbThreads;

    private final Object nbThreadWaitingLock = new Object();
    private int nbThreadWaiting;
    private boolean halt;

    public Gate(int nbThreads) {
        this.nbThreads = nbThreads;
    }

    public void waitAt() throws InterruptedException {
        synchronized (nbThreadWaitingLock) {
            nbThreadWaiting++;

            if ( nbThreadWaiting == nbThreads ) nbThreadWaitingLock.notifyAll();

            while ( nbThreadWaiting < nbThreads && !halt ) {
                try {
                    nbThreadWaitingLock.wait();
                } catch (InterruptedException e) {
                    halt = true;
                    nbThreadWaitingLock.notifyAll();
                    throw new InterruptedException();
                }
            }

            if ( halt ) {
                throw new InterruptedException();
            }
        }
    }

    public static void main(String[] args) throws InterruptedException {
        var nbThreads = 100;
        var tab = new Thread[nbThreads];

        var barrier = new Gate(nbThreads);

        IntStream.range(0, nbThreads).forEach(i -> {
            tab[i] = new Thread(() -> {
                try {
                    barrier.waitAt();
                    System.out.print(i + " ");

                } catch (InterruptedException e) {
                    return;
                }
            });
        });
        for (var i = 0; i < nbThreads-1; i++) {
            tab[i].start();
        }

        Thread.sleep(2000);
        //tab[50].interrupt();

        tab[nbThreads-1].start();
    }
}
