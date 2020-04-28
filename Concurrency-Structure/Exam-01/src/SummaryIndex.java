import java.util.Arrays;
import java.util.Random;
import java.util.Spliterator;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.DoubleBinaryOperator;

public class SummaryIndex {

    private class Task extends RecursiveTask<Double> {
        private final int from;
        private final int to;

        //private final BiFunction<T, V, V> accumulator;
        private final DoubleBinaryOperator combiner;

        private Task(int from, int to, DoubleBinaryOperator combiner) {
            this.from = from;
            this.to = to;
            this.combiner = combiner;
        }

        @Override
        protected Double compute() {
            if( this.to - this.from < 100 ) {
                return sequentialSumSummary(this.from, this.to);
            }

            int middle = this.from + (this.to - this.from)/2;
            var task1 = new Task(this.from, middle, combiner);
            var task2 = new Task(middle, this.to, combiner);
            task1.fork();
            var result2 = task2.compute();
            var result1 = task1.join();

            return combiner.applyAsDouble(result1, result2);
        }
    }

    private static class Entry {
        private double average;
        private int cursor;
        private final double[] data;

        private Entry(int dataLength) {
            this.average = Double.NaN;
            double[] data = new double[dataLength];
            Arrays.fill(data, Double.NaN);
            this.data = data;
        }
    }

    private final Entry[] entries;

    public SummaryIndex(int entryLength, int dataLength) {
        var entries = new Entry[entryLength];
        for(var i = 0; i < entries.length; i++) {
            entries[i] = new Entry(dataLength);
        }
        this.entries = entries;
    }

    public void add(int entryIndex, double value) {
        var entry = entries[entryIndex];
        var cursor = entry.cursor;
        entry.data[cursor] = value;
        entry.cursor = (cursor + 1) % entry.data.length;
    }

    public double average(int entryIndex) {  // pas utilisée dans l'exercice
        return entries[entryIndex].average;
    }

    public double sumSummary() {
        var sum = 0.0;
        for(var i = 0; i < entries.length; i++) {
            var entry = entries[i];
            var stats = Arrays.stream(entry.data).filter(v -> !Double.isNaN(v)).summaryStatistics();
            var average = stats.getAverage();;
            entry.average = average;
            if (!Double.isNaN(average)) {
                sum += stats.getSum();
            }
        }
        return sum;
    }

    double computeSumFromData(int index) {
        var entry = entries[index];
        var sum = 0.0;
        int i = 0;

        for ( ; i < entry.data.length ; i++ ) {
            if ( Double.isNaN(entry.data[i]) ) {
                break;
            }

            sum += entry.data[i];
        }

        entry.average = sum/i;

        return sum;
    }

    double sequentialSumSummary(int from, int to) {
        if ( to < from ) {
            throw new IllegalArgumentException("to needs to be greater or equals that from");
        }

        double sum = 0;

        for (var i = from ; i < to ; i++ ) {
            sum += computeSumFromData(i);
        }

        return sum;
    }

    public Double parallelSumSummary(int from, int to, DoubleBinaryOperator combiner) {
        var pool = ForkJoinPool.commonPool();
        return pool.invoke(new Task(from, to, combiner));
    }

    public static void main(String[] args) {
        var summaryIndex = new SummaryIndex(20_000, 200);

        var random = new Random(0);
        for(var i = 0; i < 10_000_000; i++) {
            summaryIndex.add(i % 20_000, random.nextInt(100));
        }

        System.out.println(summaryIndex.sumSummary());
        System.out.println(summaryIndex.sequentialSumSummary(0, summaryIndex.entries.length));
        System.out.println(summaryIndex.parallelSumSummary(0, summaryIndex.entries.length, Double::sum));

        //System.out.println(summaryIndex.parallelSumSummary());
    }
}