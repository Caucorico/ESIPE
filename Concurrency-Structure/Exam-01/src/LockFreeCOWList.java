import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;
import java.util.function.DoubleBinaryOperator;

public class LockFreeCOWList<E> {

    private E[] tab;

    private static final Object[] EMPTY = new Object[0];


    private static final VarHandle TAB_HANDLE;
    static {
        try {
            TAB_HANDLE = MethodHandles.lookup().findVarHandle(LockFreeCOWList.class, "tab", Object[].class);
        }  catch (NoSuchFieldException | IllegalAccessException e) {
            throw new AssertionError(e);
        }
    }

    @SuppressWarnings("unchecked")
    public LockFreeCOWList() {
        tab = (E[])EMPTY;
    }

    @SuppressWarnings("unchecked")
    public void add(E e) {
        Objects.requireNonNull(e);
        E[] oldArray, newArray;

        do {
            oldArray = (E[]) TAB_HANDLE.getVolatile(this);

            newArray = Arrays.copyOf(oldArray, oldArray.length + 1);
            newArray[oldArray.length] = e;
        } while ( !TAB_HANDLE.compareAndSet(this, oldArray, newArray) );
    }

    public int size() {
        return tab.length;
    }

    public static void main(String[] args) throws InterruptedException {
        var threads = new ArrayList<Thread>();
        var cowList = new LockFreeCOWList<Integer>();

        for ( int i = 0 ; i < 4 ; i++ ) {
            var thread = new Thread(() -> {
                for ( int j = 0 ; j < 2_500 ; j++ ) {
                    cowList.add(j);
                }
            });
            thread.start();
            threads.add(thread);
        }

        for ( var thread : threads ) {
            thread.join();
        }

        System.out.println("Size = " + cowList.size());
    }
}