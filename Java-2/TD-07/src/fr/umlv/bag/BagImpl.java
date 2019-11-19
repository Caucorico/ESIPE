package fr.umlv.bag;

import java.util.*;
import java.util.function.Consumer;

public class BagImpl<E> implements Bag<E>{

    private final Map<E, Integer> hashmap;
    private int size = 0;

    public BagImpl() {
        this.hashmap = new HashMap<>();
    }

    public BagImpl(Map<E, Integer> map) {
        this.hashmap = map;
    }

    @Override
    public int add(E element, int count) {
        Objects.requireNonNull(element);
        if ( count <= 0 ) throw new IllegalArgumentException("count is necessary greater than 0");
        size += count;
        return hashmap.merge(element, count, (a, b) -> a + count);
    }

    @Override
    public int count(Object element) {
        Objects.requireNonNull(element);
        return hashmap.getOrDefault(element, 0);
    }

    @Override
    public void forEach(Consumer<? super E> consumer) {
        Objects.requireNonNull(consumer);
        hashmap.forEach((key, value) -> {
            for ( var i = 0 ; i < value ; i++ ) consumer.accept(key);
        });
    }

    @Override
    public int size() {
        return this.size;
    }

    @Override
    public Iterator<E> iterator() {
            return hashmap.entrySet().stream().flatMap(m -> Collections.nCopies(m.getValue(), m.getKey()).stream()).iterator();
    }

    public Iterator<E> iteratorOld() {
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
                    remaining = entry.getValue();
                }
                remaining--;
                return current;
            }
        };
    }

}
