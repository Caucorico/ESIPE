package fr.umlv.seq;

import java.util.*;
import java.util.function.Consumer;
import java.util.stream.Collectors;

public class Seq <T> {

    private final ArrayList<T> tab;

    private Seq(ArrayList<T> list) {
        Objects.requireNonNull(list);
        tab = list;
    }

    private static <T> void throwNPEOnNullValue(List<T> list) {
            list.forEach(Objects::requireNonNull);
    }

    /**
     * List converted in ArrayList for get O(1)
     * @param list The list.
     * @param <T> Type of element.
     * @return Seq.
     */
    public static <T> Seq from(List<T> list) {
        Objects.requireNonNull(list);
        throwNPEOnNullValue(list);
        var tab = new ArrayList<T>(list);
        return new Seq<T>(tab);
    }

    public static <T> Seq of(T ... elements) {
        return from(Arrays.asList(elements));
    }

    public void forEach(Consumer<T> consumer) {
        Objects.requireNonNull(consumer);
        tab.forEach(consumer);
    }

    public T get(int i) {
        return tab.get(i);
    }

    public int size() {
        return tab.size();
    }

    @Override
    public String toString() {
        return tab.stream()
            .map(Object::toString)
            .collect(Collectors.joining(", ", "<", ">"));
    }
}
