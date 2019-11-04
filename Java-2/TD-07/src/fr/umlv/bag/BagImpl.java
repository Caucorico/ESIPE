package fr.umlv.bag;

import java.util.HashMap;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.function.Consumer;

public class BagImpl<E> implements Bag<E> {

    private final HashMap<E, Integer> hashmap = new HashMap<>();

    @Override
    public int add(E element, int count) {
        Objects.requireNonNull(element);
        if ( count <= 0 ) throw new IllegalArgumentException("count is necessary greater than 0");
        return hashmap.merge(element, count, (a, b) -> a + count);
    }

    @Override
    public int count(Object element) {
        Objects.requireNonNull(element);
        return hashmap.getOrDefault(element, 0);
    }

    @Override
    public void forEach(Consumer<E> consumer) {
        Objects.requireNonNull(consumer);
        hashmap.forEach((key, value) -> {
            for ( var i = 0 ; i < value ; i++ ) consumer.accept(key);
        });
    }

    public Iterator<E> iterator() {
        var elements = hashmap.entrySet().iterator();

        return new Iterator<>() {
            private E current;
            private int remaining;

            private final String test = "coucou";

            @Override
            public boolean hasNext() {
                return elements.hasNext() || remaining != 0;
            }

            @Override
            public E next() {
                if ( !hasNext() ) throw new NoSuchElementException();

                if ( remaining == 0 ) {
                    var entry = elements.next();
                    current = entry.getKey();
                }
                return null;
            }
        };
    }

}
