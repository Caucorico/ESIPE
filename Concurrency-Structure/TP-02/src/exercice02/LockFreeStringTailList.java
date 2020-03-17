package exercice02;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class LockFreeStringTailList {
    static final class Entry {
        private final String element;
        private volatile LockFreeStringTailList.Entry next;

        Entry(String element) {
            this.element = element;
        }
    }

    private static final VarHandle nextVarHandle;
    private static final VarHandle tailVarHandle;

    static {
        try {
            nextVarHandle = MethodHandles.lookup().findVarHandle(LockFreeStringTailList.Entry.class, "next", LockFreeStringTailList.Entry.class);
            tailVarHandle = MethodHandles.lookup().findVarHandle(LockFreeStringTailList.class, "tail", LockFreeStringTailList.Entry.class);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new AssertionError(e);
        }
    }

    private final LockFreeStringTailList.Entry head;
    private volatile LockFreeStringTailList.Entry tail;

    public LockFreeStringTailList() {
        head = new LockFreeStringTailList.Entry(null); // fake first entry
        tail = head;
    }

    public void addLast(String element) {
        Objects.requireNonNull(element);

        var entry = new LockFreeStringTailList.Entry(element);
        var oldTail = tail; // volatile read
        var last = oldTail;


        for (;;) {
            var next = last.next;
            if (next == null) {
                if(nextVarHandle.compareAndSet(last, null, entry)) {
                    tailVarHandle.compareAndSet(this, oldTail, entry);
                    return;
                }
                next = tail;
            }
            last = next;
        }
    }

    public int size() {
        var count = 0;
        for (var e = head.next; e != null; e = e.next) {
            count++;
        }
        return count;
    }

    private static Runnable createRunnable(LockFreeStringTailList list, int id) {
        return () -> {
            for (var j = 0; j < 10_000; j++) {
                list.addLast(id + " " + j);
            }
        };
    }

    public static void main(String[] args) throws InterruptedException, ExecutionException {
        var threadCount = 5;
        var list = new LockFreeStringTailList();
        var tasks = IntStream.range(0, threadCount)
                .mapToObj(id -> createRunnable(list, id))
                .map(Executors::callable)
                .collect(Collectors.toList());
        var executor = Executors.newFixedThreadPool(threadCount);
        var futures = executor.invokeAll(tasks);
        executor.shutdown();
        for(var future : futures) {
            future.get();
        }
        System.out.println(list.size());
    }
}
