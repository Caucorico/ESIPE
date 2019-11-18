package fr.umlv.xl;

import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Calc <E> {

    private final Map<String, Supplier<E>> cellsHashMap = new HashMap<>();

    public void set(String caseName, Supplier<E> supplier) {
        Objects.requireNonNull(caseName);
        Objects.requireNonNull(supplier);
        cellsHashMap.put(caseName, supplier);
    }

    public Optional<E> eval(String caseName) {
        Objects.requireNonNull(caseName);
        return Optional.ofNullable(cellsHashMap.get(caseName)).map(Supplier::get);
    }

    @Override
    public String toString() {
        return cellsHashMap.entrySet().stream()
                .map( (e) -> e.getKey() + "=" + e.getValue().get().toString())
                .collect(Collectors.joining(",", "{", "}"));
    }

    public void forEach(BiConsumer<? super String, ? super E> biConsumer) {
        cellsHashMap.entrySet().forEach( e -> biConsumer.accept(e.getKey(), e.getValue().get()));
    }

    public interface Group <E> {
        Stream<E> values();

        static <T> Group<T> of(T... elements) {
            Objects.requireNonNull(elements);
            var arrayCpy = List.copyOf(Arrays.asList(elements));
            return () -> arrayCpy.stream()
                .peek(Objects::requireNonNull);
        }

        default void forEach(Consumer<? super E> consumer) {
            values().forEach(consumer);
        }

        static Group<String> cellMatrix(int startLine, int endLine, char startColumn, char endColumn) {
            if ( startLine > endLine || startColumn > endColumn ) throw new IllegalArgumentException();

            return () -> IntStream.range(startLine, endLine+1)
                    .boxed()
                    .flatMap( l -> IntStream.range(startColumn, endColumn+1)
                    .mapToObj( c -> (char)c + "" + l));
        }

        default Group<E> ignore(Set<? super E> ignoreSet) {
            return () -> values().filter(e -> !ignoreSet.contains(e) );
        }

        default <F> Stream<F> eval(Function<? super String, Optional<F>> function) {
            /* TODO : implement this function */
            return null;
        }
    }


}
