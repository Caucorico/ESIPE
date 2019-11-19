package exercice02;

public class Counter {
    private int count;
    private final Sync<Integer> sync = new Sync<>();
    private final PermitSync<Integer> permitSync = new PermitSync<>(5);

    public int add() throws InterruptedException {
        return sync.safe(() -> count++);
    }

    public int add2() throws InterruptedException {
        return permitSync.safe(() -> count++);
    }

    public static void main(String[] args) {
        var counter = new Counter();

        Runnable runnable = () -> {
            while ( true ) {
                try {
                    Thread.sleep(10);
                    System.out.println(Thread.currentThread().getName() + " " + counter.add2() );
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        };

        for ( var i = 0 ; i < 5 ; i++ ) {
            var thread = new Thread(runnable);
            thread.setName("Thread " + i);
            thread.start();
        }
    }
}
