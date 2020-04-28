import java.util.Arrays;
import java.util.Random;
import java.util.Spliterator;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;
import java.util.function.BinaryOperator;
import java.util.function.IntBinaryOperator;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Reducer {

    private static class Task extends RecursiveTask<Integer> {
        private final int[] array;
        private final int start;
        private final int end;
        private final int initial;
        private final IntBinaryOperator op;

        Task(int start, int end, int[] array, int initial, IntBinaryOperator op) {
            this.array = array;
            this.start = start;
            this.end = end;
            this.initial = initial;
            this.op = op;
        }

        @Override
        protected Integer compute() {
            if(end - start < 1024) {
                return Arrays.stream(array ,start, end).reduce(initial, op);
            }

            var middle = (start + end) / 2 ;
            var part1 = new Task(start, middle, array, initial, op);
            var part2 = new Task(middle, end, array, initial, op);
            part1.fork();
            var result2 = part2.compute();
            var result1 = part1.join();

            return op.applyAsInt(result1, result2);
        }
    }

    public static int sum(int[] array) {
        /*var sum = 0;
        for(var value: array) {
            sum += value;
        }
        return sum;*/

        return reduce(array, 0, Integer::sum);
    }

    public static int max(int[] array) {
        /*var max = Integer.MIN_VALUE;
        for(var value: array) {
            max = Math.max(max, value);
        }
        return max;*/

        return reduce(array, Integer.MIN_VALUE, Math::max);
    }

    public static int reduce(int[] array, int initial, IntBinaryOperator op) {
        var acc = initial;
        for ( var value : array ) {
            acc = op.applyAsInt(acc, value);
        }

        return acc;
    }

    public static int reduceWithStream(int[] s, int initial, IntBinaryOperator op) {
        return Arrays.stream(s).reduce(initial, op);
    }

    public static int parallelReduceWithStream(int[] s, int initial, IntBinaryOperator op) {
        return Arrays.stream(s).parallel().reduce(initial, op);
    }

    public static int parallelReduceWithForkJoin(int[] s, int initial, IntBinaryOperator op) {
        /* Il ne faut pas utiliser de ThreadPoolExecutor parce que nous ne pouvons pas faire de join dans les des Callable pour éviter des Deadlock */
        /* Or, dans notre cas, il faut des join, car si la tâche est trop grosse, on fork() et du coup on doit nécessairement faire un join */
        /* C'est pourquoi il faut utiliser un ForkJoinPool. */

        /* Par défaut, pour obtenir un ForkJoinPool, on fait : */
        ForkJoinPool commonPool = ForkJoinPool.commonPool();

        return commonPool.invoke(new Task(0, s.length, s, initial, op));

    }

    public static <T> T sequentialReduce(Spliterator<T> spliterator, T initial, BinaryOperator<T> op) {
        var box = new Object() {
            private T acc = initial;
        };
        while (spliterator.tryAdvance(e -> {
            box.acc = op.apply(box.acc, e);
        }));

        return box.acc;
    }

    public static void main(String[] args) {
        var random = new Random(20);
        var ints = random.ints(1_000_000, 0, 1_000).toArray();
        System.out.println(max(ints));
        System.out.println(reduceWithStream(ints, Integer.MIN_VALUE, Math::max));
        System.out.println(parallelReduceWithStream(ints, Integer.MIN_VALUE, Math::max));
        System.out.println(parallelReduceWithForkJoin(ints, Integer.MIN_VALUE, Math::max));
        System.out.println(sum(ints));
        System.out.println(reduceWithStream(ints, 0, Integer::sum));
        System.out.println(parallelReduceWithStream(ints, 0, Integer::sum));
        System.out.println(parallelReduceWithForkJoin(ints, 0, Integer::sum));
    }
}