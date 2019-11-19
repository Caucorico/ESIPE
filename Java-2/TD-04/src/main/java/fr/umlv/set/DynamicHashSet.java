package fr.umlv.set;

import java.util.Objects;
import java.util.function.Consumer;

public class DynamicHashSet <E> {

    private Entry<E>[] hashTable;

    private int size;

    private int currentSize;

    public DynamicHashSet() {
        this.size = 8;
        this.currentSize = 0;
        this.hashTable = (Entry<E>[]) new Entry[this.size];
    }

    public DynamicHashSet(int size) {
        this.size = size;
        this.hashTable = new Entry[this.size];
    }

    private int hashFunction( E value ) {
        return value.hashCode()&(size-1);
    }

    private void reSize() {
        if ( currentSize >= size/2 ) {
            var newTable = (Entry<E>[]) new Entry[size*2];
            forEach( e -> genericAdd(e, newTable));
            hashTable = newTable;
            size *= 2;
        }
    }


    public void add( E value ) {
        reSize();
        if ( genericAdd(value, hashTable) ) currentSize++;
    }

    private boolean genericAdd( E value, Entry<E>[] localHashTable ) {
        var hash = hashFunction(value);
        if ( hashContains(value, hash, localHashTable) ) return false;
        localHashTable[hash] = new Entry<>(value, localHashTable[hash]);
        return true;
    }

    public int size() {
        return currentSize;
    }

    public void forEach(Consumer<? super E> c) {
        for ( int i = 0 ; i < size ; i++ ) {
            var current = hashTable[i];
            while ( current != null ) {
                c.accept(current.value);
                current = current.next;
            }
        }
    }

    public boolean hashContains( Object value, int hash, Entry<E>[] localHashTable ) {
        var current = localHashTable[hash];
        while ( current != null ) {
            if ( current.value.equals(value) ) return true;
            current = current.next;
        }
        return false;
    }

    public boolean contains( Object value ) {
        Objects.requireNonNull(value);
        for ( int i = 0 ; i < size ; i++ ) {
            if ( hashContains(value, i, hashTable) ) return true;
        }
        return false;
    }

    static class Entry <E> {

        private E value;

        private Entry next;

        public Entry( E value ) {
            this.value = value;
            this.next = null;
        }

        public Entry( E value, Entry next ) {
            this.value = value;
            this.next = next;
        }
    }
}
