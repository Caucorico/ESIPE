package exercice03;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

public class COWSet<E> {
    private final E[][] hashArray;

    private static final Object[] EMPTY = new Object[0];

    private static final VarHandle hashArrayElementsHandle;

    static {
        hashArrayElementsHandle = MethodHandles.arrayElementVarHandle(Object[][].class);
    }

    @SuppressWarnings("unchecked")
    public COWSet(int capacity) {
        var array = new Object[capacity][];
        Arrays.setAll(array, __ -> EMPTY);
        this.hashArray = (E[][])array;
    }

    @SuppressWarnings("unchecked")
    public boolean add(E element) {
        Objects.requireNonNull(element);
        var index = element.hashCode() % hashArray.length;
        E[] oldArray, newArray;

        do {
            oldArray = (E[]) hashArrayElementsHandle.getVolatile(hashArray, index);

            for (var e : oldArray) {
                if (element.equals(e)) {
                    return false;
                }
            }

            newArray = Arrays.copyOf(oldArray, oldArray.length + 1);
            newArray[oldArray.length] = element;
        } while ( !hashArrayElementsHandle.compareAndSet(hashArray, index, oldArray, newArray));

        return true;
    }

    @SuppressWarnings("unchecked")
    public void forEach(Consumer<? super E> consumer) {
        for(var index = 0; index < hashArray.length; index++) {
            var oldArray = (E[])hashArrayElementsHandle.getVolatile(hashArray, index);
            for(var element: oldArray) {
                consumer.accept(element);
            }
        }
    }

    public static void main(String[] args) throws InterruptedException {
        COWSet<Integer> cowSet = new COWSet<>(8);
        int thread_number = 4;
        Thread[] threads = new Thread[thread_number];


        for ( int i = 0 ; i < thread_number ; i++ ) {
            Thread t = new Thread(() -> {
                for ( int j = 0 ; j < 10_000 ; j++ ) {
                    cowSet.add(j);
                }
            });
            threads[i] = t;
            t.start();
        }

        for ( int i = 0 ; i < thread_number ; i++ ) {
            threads[i].join();
        }

        AtomicInteger sum = new AtomicInteger(0);

        Consumer<Integer> consumer = element -> {
            sum.incrementAndGet();
        };

        cowSet.forEach(consumer);

        System.out.println(sum.get());
    }
}