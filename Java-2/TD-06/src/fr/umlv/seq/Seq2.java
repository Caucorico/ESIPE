package fr.umlv.seq;

import java.util.*;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

public class Seq2<T> implements Iterable<T> {

    private final ArrayList<Object> tab;

    private final Function<Object,? extends T> applier;

    @SuppressWarnings("unchecked")
    private Seq2(ArrayList<Object> list) {
        this(list, (e) -> (T) e);
    }

    private Seq2(ArrayList<Object> list, Function<Object, ? extends T> function) {
        Objects.requireNonNull(list);
        tab = list;
        applier = function;
    }

    private static <T> void throwNPEOnNullValue(List<T> list) {
            list.forEach(Objects::requireNonNull);
    }

    /**
     * List converted in ArrayList for get O(1)
     * @param list The list.
     * @return Seq.
     */
    public static <R> Seq2<R> from(List<? extends R> list) {
        Objects.requireNonNull(list);
        throwNPEOnNullValue(list);
        var tab = new ArrayList<Object>(list);
        return new Seq2<>(tab);
    }

    public static <T> Seq2 of(T ... elements) {
        return from(Arrays.asList(elements));
    }

    public void forEach(Consumer<? super T> consumer) {
        Objects.requireNonNull(consumer);
        tab.stream().map(applier).forEach(consumer);
    }

    public T get(int i) {
        return applier.apply(tab.get(i));
    }

    public int size() {
        return tab.size();
    }

    public <R> Seq2<R> map(Function<? super T, ? extends R> function) {
        Objects.requireNonNull(function);
        return new Seq2<>(tab, applier.andThen(function));
    }

    @Override
    public String toString() {
        return tab.stream()
            .map(applier)
            .map(Object::toString)
            .collect(Collectors.joining(", ", "<", ">"));
    }

    public Optional<? extends T> findFirst() {
        return tab.stream()
                .map(applier)
                .findFirst();
    }

    @Override
    public Iterator<T> iterator() {
        return new Iterator<T>() {

            private int i;

            @Override
            public boolean hasNext() {
                return i < tab.size();
            }

            @Override
            public T next() {
                if ( !hasNext() ) throw new NoSuchElementException();
                return get(i++);
            }
        };
    }

    public Stream<T> stream() {
        var spliterator = Spliterators.spliterator(iterator(), size(), Spliterator.ORDERED|Spliterator.IMMUTABLE|Spliterator.NONNULL );
        return StreamSupport.stream(spliterator, false);
    }
}
