package fr.umlv.queue;

import java.util.*;

public class ResizeableFifo<E> extends AbstractQueue<E> {
    private int size;

    private int maxSize;

    private int tail; /* pos of the newest element */

    private int head; /* pos of the oldest element */

    private E[] tab;

    @SuppressWarnings("unchecked")
    public ResizeableFifo(int maxSize) {
        if ( maxSize <= 0 ) {
            throw new IllegalArgumentException("The size cannot be negative or ''null''");
        }
        this.size = 0;
        this.tail = -1;
        this.head = 0;
        this.maxSize = maxSize;
        tab = (E[]) new Object[maxSize];
    }

    @Override
    public boolean isEmpty() {
        return size == 0;
    }

    @Override
    public int size() { return size; }

    @SuppressWarnings("unchecked")
    private void resize() {
        if ( size < maxSize ) return;
        var buffTab = (E[]) new Object[maxSize];
        buffTab = Arrays.copyOf(buffTab, maxSize*2);
        System.arraycopy(tab, head, buffTab, 0, maxSize-head);
        System.arraycopy(tab, 0, buffTab, maxSize-head, tail+1);
        tab = buffTab;

        head = 0;
        tail = maxSize-1;
        maxSize *= 2;
    }

    private int incrementPos(int pos) {
        if ( pos+1 >= maxSize ) {
            pos = 0;
        } else {
            pos++;
        }
        return pos;
    }

    private int incrementSize(int size) {
        if ( size >= maxSize ) return size;
        return ++size;
    }

    private int decrementSize(int size) {
        if ( size <= 0 ) return size;
        else return --size;
    }

    @Override
    public boolean offer(E element) {
        Objects.requireNonNull(element);
        resize();
        tail = incrementPos(tail);
        /*
        Enable this condition to overwrite head value instead of error
        if ( size == maxSize ) head = incrementPos(head);
        */
        tab[tail] = element;
        size = incrementSize(size);

        return true;
    }

    @Override
    public E poll() {
        //if ( isEmpty() ) throw new IllegalStateException("The queue is empty !");
        if ( isEmpty() ) return null;
        var buff = tab[head];
        head = incrementPos(head);
        size = decrementSize(size);
        return buff;
    }

    @Override
    public E peek() {
        return tab[head];
    }

    public String toStringState() {
        StringJoiner sj = new StringJoiner(", ", "[", "]");

        for ( int i = 0 ; i < size ; i++ ) {
            sj.add(tab[i].toString());
        }

        return sj.toString();
    }

    @Override
    public String toString() {
        StringJoiner sj = new StringJoiner(", ", "[", "]");

        for ( int i = head, j = 0  ; j < size ; i = incrementPos(i), j++ ) {
            sj.add(tab[i].toString());
        }

        return sj.toString();
    }

    public static void main(String[] args) {
        Fifo<Integer> fifo = new Fifo<>(10);
        for ( int i = 0 ; i < 15 ; i++ ) {
            fifo.offer(i);
        }

        System.out.println(fifo.toString());

        for ( int i = 0 ; i < 3 ; i++ ) {
            fifo.poll();
        }
        System.out.println(fifo.toString());
    }

    @Override
    public Iterator<E> iterator() {
        return new Iterator<E>() {
            private int index = head;
            private int iteration;

            @Override
            public boolean hasNext() {
                return this.iteration < size;
            }

            @Override
            public E next() {
                int buff = this.index;
                if ( !hasNext() ) throw new NoSuchElementException("No next available !");
                this.index = incrementPos(this.index);
                this.iteration++;
                return tab[buff];
            }
        };
    }
}
