package fr.umlv.conc;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;

public class ReentrantSpinLock {
    private volatile int lock;
    private /*volatile*/ Thread ownerThread;

    private static final VarHandle HANDLE;

    static {
        try {
            HANDLE = MethodHandles.lookup().findVarHandle(ReentrantSpinLock.class, "lock", int.class);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new AssertionError(e);
        }
    }

    public void lock() {
        var current = Thread.currentThread();
        for (;;){
            if (HANDLE.compareAndSet(this, 0, 1)){
                this.ownerThread = current;
                return;
            }
            if (ownerThread == current){
                lock++;
                return;
            }
            Thread.onSpinWait();
        }
    }

    public void unlock() {
        if (ownerThread != Thread.currentThread()) {
            throw new IllegalStateException();
        }

        var l = this.lock; // lecture volatile
        if (l == 1) {
            ownerThread = null;
            this.lock = 0; // ecriture volatile
            return;
        }
        this.lock = l - 1; // ecriture volatile
    }

    public static void main(String[] args) throws InterruptedException {
        var runnable = new Runnable() {
            private int counter;
            private final ReentrantSpinLock spinLock = new ReentrantSpinLock();

            @Override
            public void run() {
                for(var i = 0; i < 1_000_000; i++) {
                    spinLock.lock();
                    try {
                        spinLock.lock();
                        try {
                            counter++;
                        } finally {
                            spinLock.unlock();
                        }
                    } finally {
                        spinLock.unlock();
                    }
                }
            }
        };
        var t1 = new Thread(runnable);
        var t2 = new Thread(runnable);
        t1.start();
        t2.start();
        t1.join();
        t2.join();
        System.out.println("counter " + runnable.counter);
    }
}
