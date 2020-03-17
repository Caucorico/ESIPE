package exercice03;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.Arrays;
import java.util.Objects;
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
            hashArray[index] = newArray;
        } while ( !hashArrayElementsHandle.compareAndSet(hashArray, index, oldArray, newArray));

        return true;
    }

    public void forEach(Consumer<? super E> consumer) {
        for(var index = 0; index < hashArray.length; index++) {
            var oldArray = hashArray[index];
            for(var element: oldArray) {
                consumer.accept(element);
            }
        }
    }
}