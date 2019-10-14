package fr.umlv.set;

import java.util.function.Consumer;

public class DynamicHashSet {

    private Entry[] hashTable;

    private int size;

    private int currentSize;

    public DynamicHashSet() {
        this.size = 8;
        this.currentSize = 0;
        this.hashTable = new Entry[this.size];
    }

    public DynamicHashSet(int pow) {
        this.size = (int)Math.pow(2, pow);
        this.hashTable = new Entry[this.size];
    }

    private int hashFunction( int value ) {
        //return value&0x1;
        return value&(size-1);
    }

    public void add( int value ) {
        var hash = hashFunction(value);
        var current = hashTable[hash];
        while ( current != null ) {
            if ( current.value == value ) return;
            current = current.next;
        }
        hashTable[hash] = new Entry(value, hashTable[hash]);
        currentSize++;
    }

    public int size() {
        return currentSize;
    }

    public void forEach(Consumer<Integer> c) {
        for ( int i = 0 ; i < size ; i++ ) {
            var current = hashTable[i];
            while ( current != null ) {
                c.accept(current.value);
                current = current.next;
            }
        }
    }

    public boolean contains( int value ) {
        for ( int i = 0 ; i < size ; i++ ) {
            var current = hashTable[i];
            while ( current != null ) {
                if ( current.value == value ) return true;
                current = current.next;
            }
        }
        return false;
    }

    static class Entry {

        private int value;

        private Entry next;

        public Entry( int value ) {
            this.value = value;
            this.next = null;
        }

        public Entry( int value, Entry next ) {
            this.value = value;
            this.next = next;
        }

    }
}
