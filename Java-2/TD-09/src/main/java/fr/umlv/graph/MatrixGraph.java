package fr.umlv.graph;

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.IntStream;

class MatrixGraph <T> implements Graph<T> {

    private final T[] tab;

    private final int base;

    @SuppressWarnings("unchecked")
    MatrixGraph(int nodeNumber) {
        if ( nodeNumber < 0 ) {
            throw new IllegalArgumentException("nodeNumber cannot be negative");
        }

        this.base = nodeNumber;
        this.tab = (T[]) new Object[this.base*this.base];
    }

    private int index(int src, int dst) {
        if ( src >= base || src < 0  ) {
            throw new IndexOutOfBoundsException("src : " + src + " index out of bounds");
        } else if ( dst >= base || dst < 0 ) {
            throw new IndexOutOfBoundsException("dst : " + dst + " index out of bounds");
        }

        return src*base + dst;
    }

    private int getNextValidEdge(int src, int dst) {
        var i = dst;
        while ( i < base ) {
            if ( getWeight(src, i).isPresent() ) return i;
            i++;
        }

        return -1;
    }


    @Override
    public Optional<T> getWeight(int src, int dst) {
        return Optional.ofNullable(tab[index(src, dst)]);
    }

    @Override
    public void addEdge(int src, int dst, T weight) {
        Objects.requireNonNull(weight);
        tab[index(src, dst)] = weight;
    }

    @Override
    public void edges(int src, EdgeConsumer<? super T> consumer) {
        Objects.requireNonNull(consumer);

        IntStream.range(0, base-1)
            .forEach( dst -> {
                var optional = getWeight(src, dst);
                optional.ifPresent(o -> consumer.edge(src, dst, o));
            });
    }

    @Override
    public Iterator<Integer> neighborIterator(int src) {
        return new Iterator<Integer>() {
            int lastIndex = -1;
            int i = getNextValidEdge(src, 0);

            @Override
            public boolean hasNext() {
                return i >= 0;
            }

            @Override
            public Integer next() {
                var index = i;
                if ( index < 0 ) throw new NoSuchElementException();
                i = getNextValidEdge(src, i+1);
                lastIndex = index;
                return index;
            }

            @Override
            public void remove() {
                if ( lastIndex == -1 || lastIndex == i ) throw new IllegalStateException();
                addEdge(src, lastIndex, null);
            }
        };
    }


}
