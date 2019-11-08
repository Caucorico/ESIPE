package fr.umlv.bag;

import java.util.*;
import java.util.function.Consumer;

public interface Bag<E> extends Iterable<E>{

    int add(E element, int count);

    int count(Object element);

    void forEach(Consumer<? super E> consumer);

    static <T> BagImpl<T> createSimpleBag() {
        return new BagImpl<>();
    }

    int size();

    Iterator<E> iterator();

    default Collection<E> asCollection() {
         var bag = this;
         return new AbstractCollection<E>() {
             @Override
             public Iterator<E> iterator() {
                 return bag.iterator();
             }

             @Override
             public int size() {
                 return bag.size();
             }

             @Override
             public boolean contains(Object o) {
                 return bag.count(o) != 0;
             }
         };
    };

    static <F> Bag<F>  createOrderedByInsertionBag() {
        return new BagImpl<>(new LinkedHashMap<>());
    }

    static <F> Bag<F> createOrderedByElementBag(Comparator<? super F> comparator) {
        return new BagImpl<>(new TreeMap<>(comparator));
    }

    static <F extends Comparable<? super F>> Bag<F> createOrderedByElementBagFromCollection ( Collection<? extends F> collection) {
        var bag = createOrderedByElementBag(Comparator.<F>naturalOrder());
        collection.forEach(element -> bag.add(element, 1));
        return bag;
    }

}
