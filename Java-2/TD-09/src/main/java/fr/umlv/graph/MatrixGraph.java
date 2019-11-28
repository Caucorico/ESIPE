package fr.umlv.graph;

import java.util.*;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;
import java.util.stream.StreamSupport;

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

    public int getNextNeighborIndex(int src, int startDst) {
        for ( var i = startDst ; i < base ; i++ ) {
            if ( getWeight(src, i).isPresent() ) return i;
        }
        return -1;
    }

    @Override
    public Iterator<Integer> neighborIterator(int src) {
        return new Iterator<Integer>() {

            /* In the iterator, the current element begin at the first neighbor */
            int index = getNextNeighborIndex(src, 0);

            int lastReturnedIndex = -1;

            @Override
            public boolean hasNext() {
                if ( index < 0 ) return false;
                return true;
            }

            @Override
            public Integer next() {
                if ( !hasNext() ) throw new NoSuchElementException();
                lastReturnedIndex = index;
                index = getNextNeighborIndex(src, index+1);
                return lastReturnedIndex;
            }

            @Override
            public void remove() {
                if ( lastReturnedIndex == -1 ) throw new IllegalStateException();
                tab[index(src, lastReturnedIndex)] = null;
                lastReturnedIndex = -1;
            }
        };
    }

    @Override
    public IntStream neighborStream(int src) {
        return StreamSupport.intStream(new Spliterator.OfInt() {
            @Override
            public OfInt trySplit() {
                return null;
            }

            @Override
            public boolean tryAdvance(IntConsumer action) {
                return false;
            }

            @Override
            public long estimateSize() {
                return 0;
            }

            @Override
            public int characteristics() {
                return 0;
            }
        }, true);
    }
}
