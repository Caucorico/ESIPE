package fr.umlv.bag;

import java.util.function.Consumer;

public interface Bag<E> {

    int add(E element, int count);

    int count(Object element);

    void forEach(Consumer<E> consumer);

    static <T> BagImpl<T> createSimpleBag() {
        return new BagImpl<>();
    }

}
