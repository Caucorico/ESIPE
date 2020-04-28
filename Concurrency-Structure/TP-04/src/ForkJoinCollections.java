import java.util.Collection;
import java.util.Spliterator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ForkJoinCollections {
    public static <V, T> V forkJoinReduce(Collection<T> collection, int threshold, V initialValue,
                                          XXX accumulator, YYY combiner) {

        return forkJoinReduce(collection.spliterator(), threshold, initialValue, accumulator, combiner);
    }

    private static <V, T> V forkJoinReduce(Spliterator<T> spliterator, int threshold, V initialValue,
                                           XXX accumulator, YYY combiner) {
        // TODO
    }

    public static void main(String[] args) {
        // sequential
        System.out.println(IntStream.range(0, 10_000).sum());

        // fork/join
        var list = IntStream.range(0, 10_000).boxed().collect(Collectors.toList());
        var result = forkJoinReduce(list, 1_000, 0, (acc, value) -> acc + value, (acc1, acc2) -> acc1 + acc2);
        System.out.println(result);
    }
}
