package exercice02;

import java.util.HashMap;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Supplier;

public class PermitSync<V> {

    private final ReentrantLock sealKeeper = new ReentrantLock(false);

    private final HashMap<Supplier, CongestionLock> reentrantMap = new HashMap<>();

    private final int permits;

    public PermitSync(int permits) {
        this.permits = permits;
    }

    private CongestionLock getCongestionLockBySupplier(Supplier<? extends V> supplier) {
        sealKeeper.lock();
        try {
            reentrantMap.putIfAbsent(supplier, new CongestionLock());
            return reentrantMap.get(supplier);
        } finally {
            sealKeeper.unlock();
        }
    }

    public V safe(Supplier<? extends V> supplier) throws InterruptedException {
        var congestionLock = getCongestionLockBySupplier(supplier);
        congestionLock.enter();
        try {
            return supplier.get();
        } finally {
            congestionLock.leave();
        }
    }

    private class CongestionLock {
        private final ReentrantLock lock = new ReentrantLock(false);
        private final Condition maxGuest = lock.newCondition();
        private int guest;

        public int enter() throws InterruptedException {
            lock.lock();
            try {
                while ( guest >= permits ) maxGuest.await();
                return ++guest;
            } finally {
                lock.unlock();
            }
        }

        public int leave() {
            lock.lock();
            try {
                maxGuest.signalAll();
                return --guest;
            } finally {
                lock.unlock();
            }
        }
    }
}
