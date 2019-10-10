package exercice01;

import java.util.ArrayList;

public class ThreadSafeList<T> {
    private ArrayList<T> arrayList;
    private final Object lock = new Object();

    public ThreadSafeList() {
        this.arrayList = new ArrayList<>();
    }

    public ThreadSafeList(int size) {
        this.arrayList = new ArrayList<>(size);
    }

    public boolean add(T element) {
        synchronized (lock) {
            return arrayList.add(element);
        }
    }

    public int size() {
        synchronized (lock) {
            return arrayList.size();
        }
    }

    @Override
    public String toString() {
        synchronized (lock) {
            return arrayList.toString();
        }
    }
}
