package exercice01;

import java.util.ArrayList;
import java.util.stream.IntStream;

public class HelloListBugFixed {
        public static void main(String[] args) throws InterruptedException {
            final var lock = new Object();
            var nbThreads = 4;
            var threads = new Thread[nbThreads];

            var list = new ArrayList<Integer>();

            IntStream.range(0, nbThreads).forEach(j -> {
                Runnable runnable = () -> {
                    for (var i = 0; i < 5000; i++) {
                        synchronized (lock) {
                            list.add(i);
                        }
                    }
                };

                threads[j] = new Thread(runnable);
                threads[j].start();
            });

            for (Thread thread : threads) {
                thread.join();
            }

            System.out.println("taille de la liste:" + list.size());
        }
}
