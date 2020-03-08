package exercice03;

import exercice02.Counter;

import java.util.ArrayList;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

public class Linked<E> {
    private static class Entry<E> {
        private final E element;
        private final Entry<E> next;

        private Entry(E element, Entry<E> next) {
            this.element = element;
            this.next = next;
        }
    }

    private final AtomicReference<Entry<E>> head = new AtomicReference<>();

    public void addFirst(E element) {
        Objects.requireNonNull(element);
        Entry<E> head;
        do { head = this.head.get(); }
        while ( !this.head.compareAndSet(head, new Entry<>(element, head)) );
    }

    public int size() {
        var size = 0;
        for(var link = head.get(); link != null; link = link.next)  {
            size ++;
        }
        return size;
    }

    public static void main(String[] args) throws InterruptedException {
        Linked<Integer> list = new Linked<>();
        ArrayList<Thread> threads = new ArrayList<>();

        for ( var i = 0 ; i < 4 ; i++ ) {
            var thread = new Thread(() -> {
                for ( var j = 0 ; j < 10_000 ; j++ ) {
                    list.addFirst(j);
                }
            });

            thread.start();
            threads.add(thread);
        }

        for ( var thread : threads ) {
            thread.join();
        }

        System.out.println(list.size());
    }
}
