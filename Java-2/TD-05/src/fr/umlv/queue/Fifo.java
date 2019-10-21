package fr.umlv.queue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Objects;
import java.util.StringJoiner;

public class Fifo <E> {

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
        tail = incrementPos(tail);
        if ( size == maxSize ) head = incrementPos(head);
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
}
