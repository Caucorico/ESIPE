package exercice03;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.Consumer;

public class COWSet<E> {
    private final E[][] hashArray;

    private static final Object[] EMPTY = new Object[0];

    @SuppressWarnings("unchecked")
    public COWSet(int capacity) {
        var array = new Object[capacity][];
        Arrays.setAll(array, __ -> EMPTY);
        this.hashArray = (E[][])array;
    }

    public boolean add(E element) {
        Objects.requireNonNull(element);
        var index = element.hashCode() % hashArray.length;
        for (var e : hashArray[index]) {
            if (element.equals(e)) {
                return false;
            }
        }
        var oldArray = hashArray[index];
        var newArray = Arrays.copyOf(oldArray, oldArray.length + 1);
        newArray[oldArray.length] = element;
        hashArray[index] = newArray;
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