package exercice01;

public class Exchanger<T> {
    private final Object lock = new Object();
    private States state;
    private T value1;
    private T value2;

    public Exchanger() {
        synchronized (lock) {
            this.state = States.EMPTY;

        }
    }

    public T exchange(T value) throws InterruptedException {
        synchronized (lock) {
            if ( state == States.MED ) {
                value2 = value;
                state = States.FULL;
                lock.notify();
                return value1;
            } else {
                value1 = value;
                state = States.MED;
                while ( state != States.FULL) {
                    lock.wait();
                }
                this.state = States.EMPTY;
                return value2;
            }
        }
    }

    private enum States { EMPTY, MED, FULL };

}
