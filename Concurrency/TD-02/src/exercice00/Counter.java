package exercice00;

public class Counter {

        private int value;

        public void add10000() {
            for (var i = 0; i < 10_000; i++) {
                value++;
            }
        }

        public static void main(String[] args) throws InterruptedException {
            var counter = new Counter();
            var thread1 = new Thread(counter::add10000);
            var thread2 = new Thread(counter::add10000);
            thread1.start();
            thread2.start();
            thread1.join();
            thread2.join();
            System.out.println(counter.value);
        }
    }

