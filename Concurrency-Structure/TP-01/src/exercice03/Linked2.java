package exercice03;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.ArrayList;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

public class Linked2<E> {
    private static class Entry<E> {
        private final E element;
        private final Entry<E> next;

        private Entry(E element, Entry<E> next) {
            this.element = element;
            this.next = next;
        }
    }

    private volatile Entry<E> head;
    private static final VarHandle headHandle;

    static {
        try {
            headHandle = MethodHandles.lookup().findVarHandle(Linked2.class, "head", Entry.class);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new Error(e);
        }
    }

    public void addFirst(E element) {
        Objects.requireNonNull(element);
        Entry<E> head;
        do {
            head = this.head;
        } while ( !headHandle.compareAndSet(this, head, new Entry<>(element, head)));
    }

    @SuppressWarnings("unchecked")
    public int size() {
        var size = 0;
        for(var link = (Entry<E>)headHandle.get(this) ; link != null; link = link.next) {
            size ++;
        }
        return size;
    }

    public static void main(String[] args) throws InterruptedException {
        ArrayList<Thread> threads = new ArrayList<>();
        Linked2<Integer> linked2 = new Linked2<>();

        for ( var i = 0 ; i < 4 ; i++ ) {
            var thread = new Thread(() -> {
               for ( var j = 0 ; j < 1_000 ; j++ ) {
                   linked2.addFirst(j);
               }
            });

            threads.add(thread);
            thread.start();
        }

        for ( var i = 0 ; i < 4 ; i++ ) {
            threads.get(i).join();
        }

        System.out.println(linked2.size());
    }
}
