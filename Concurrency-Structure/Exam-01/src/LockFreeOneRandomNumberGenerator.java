import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;

/**
 * Le RandomNumberGenerator n'est pas threadsafe.
 * En effet, dans la méthode next, l'opérande ^= est utilisé. Or, cet opérande n'est pas atomique.
 * (pour x ^= x)On peut le décomposer en :
 *  local = x^x
 *  x = local
 *
 *  Et étant donné que le programme peut s'interrompre entre ses deux lignes, il va y avoir des problèmes.
 */

public class LockFreeOneRandomNumberGenerator {
    private long x;

    private static final VarHandle TAB_HANDLE;
    static {
        try {
            TAB_HANDLE = MethodHandles.lookup().findVarHandle(LockFreeOneRandomNumberGenerator.class, "x", long.class);
        }  catch (NoSuchFieldException | IllegalAccessException e) {
            throw new AssertionError(e);
        }
    }

    public LockFreeOneRandomNumberGenerator(long seed) {
        if (seed == 0) {
            throw new IllegalArgumentException("seed == 0");
        }
        x = seed;
    }

    public long next() {  // Marsaglia's XorShift
        long old_x, new_x;
        do {
            old_x = x;
            new_x = old_x;
            new_x ^= new_x >>> 12;
            new_x ^= new_x << 25;
            new_x ^= new_x >>> 27;
        } while (!TAB_HANDLE.compareAndSet(this, old_x, new_x) );

        return x * 2685821657736338717L;
    }

    public static void main(String[] args) {
        var rng = new RandomNumberGenerator(1);
        for(var i = 0; i < 5_000; i++) {
            System.out.println(rng.next());
        }
    }
}
