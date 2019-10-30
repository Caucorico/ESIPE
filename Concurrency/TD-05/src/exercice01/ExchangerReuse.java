package exercice01;

public class ExchangerReuse<T> {
    private final Object lock = new Object();
    private States state;
    private T value1;
    private T value2;

    public ExchangerReuse() {
        synchronized (lock) {
            this.state = States.EMPTY;

        }
    }

    public T exchange(T value) throws InterruptedException {
        synchronized (lock) {
            while (state == States.FULL) {
                lock.wait();
            }
            if ( state == States.MED ) {
                value2 = value;
                state = States.FULL;
                lock.notifyAll();
                return value1;
            } else {
                value1 = value;
                state = States.MED;
                while ( state != States.FULL) {
                    lock.wait();
                }
                state = States.EMPTY;
                lock.notifyAll();
                return value2;
            }
        }
    }

    private enum States { EMPTY, MED, FULL };

}
