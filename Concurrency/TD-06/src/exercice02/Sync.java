package exercice02;

import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Supplier;

public class Sync<V> {
    private boolean inSafe;
    private final ReentrantLock safeLock = new ReentrantLock(false);
    private final ReentrantLock inSafeLock = new ReentrantLock(false);

    public boolean inSafe() {
        inSafeLock.lock();
        try {
            return inSafe;
        } finally {
            inSafeLock.unlock();
        }
    }

    public V safe(Supplier<? extends V> supplier) throws InterruptedException {
        safeLock.lock();
        try {
            inSafeLock.lock();
            try {
                inSafe = true;
            } finally {
                inSafeLock.unlock();
            }
            return supplier.get();
        } finally {
            safeLock.unlock();
            inSafeLock.lock();
            try {
                inSafe = false;
            } finally {
                inSafeLock.unlock();
            }
        }

    }
}
