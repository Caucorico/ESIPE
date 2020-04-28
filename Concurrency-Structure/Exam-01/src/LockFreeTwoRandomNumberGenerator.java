import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;

public class LockFreeTwoRandomNumberGenerator {
    private long x;

    private static final VarHandle TAB_HANDLE;
    static {
        try {
            TAB_HANDLE = MethodHandles.lookup().findVarHandle(LockFreeTwoRandomNumberGenerator.class, "x", long.class);
        }  catch (NoSuchFieldException | IllegalAccessException e) {
            throw new AssertionError(e);
        }
    }

    public LockFreeTwoRandomNumberGenerator(long seed) {
        if (seed == 0) {
            throw new IllegalArgumentException("seed == 0");
        }
        x = seed;
    }

    public long next() {  // Marsaglia's XorShift
        long  new_x, old_x;
        while (true) {

            old_x = x;
            new_x = old_x;
            new_x ^= new_x >>> 12;
            new_x ^= new_x << 25;
            new_x ^= new_x >>> 27;

            var res = (Long)TAB_HANDLE.getAndSet(this, new_x);
            if ( res == old_x ) break;
        }

        return x * 2685821657736338717L;
    }

    public static void main(String[] args) {
        var rng = new RandomNumberGenerator(1);
        for(var i = 0; i < 5_000; i++) {
            System.out.println(rng.next());
        }
    }
}
