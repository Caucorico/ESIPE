package exercice02;

import java.util.function.Supplier;

public class Sync<V> {
    public boolean inSafe() {
        // TODO
    }

    public V safe(Supplier<? extends V> supplier) throws InterruptedException {
        return supplier.get();  // TODO
    }
}
