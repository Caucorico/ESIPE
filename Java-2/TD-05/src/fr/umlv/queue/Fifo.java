package fr.umlv.queue;

import java.util.*;

public class Fifo <E> implements Iterable<E>{

    private int size;

    private int maxSize;

    private int tail;

    private int head;

    private E[] tab;

    @SuppressWarnings("unchecked")
    public Fifo(int maxSize) {
        if ( maxSize <= 0 ) {
            throw new IllegalArgumentException("The size cannot be negative or ''null''");
        }
        this.size = 0;
        this.tail = -1;
        this.head = 0;
        this.maxSize = maxSize;
        tab = (E[]) new Object[maxSize];
    }

    public boolean isEmpty() {
        return size == 0;
    }

    public int size() { return size; }

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

    public void offer(E element) {
        Objects.requireNonNull(element);
        if ( size == maxSize ) throw new IllegalStateException("The stack is full !!!");
        tail = incrementPos(tail);
        /*
        Enable this condition to overwrite head value instead of error
        if ( size == maxSize ) head = incrementPos(head);
        */
        tab[tail] = element;
        size = incrementSize(size);
    }

    public E poll() {
        if ( isEmpty() ) throw new IllegalStateException("The queue is empty !");
        var buff = tab[head];
        head = incrementPos(head);
        size = decrementSize(size);
        return buff;
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
        Fifo<Integer> fifo = new Fifo<>(30);
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
