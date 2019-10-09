package exercice03;

import java.util.ArrayList;
import java.util.stream.IntStream;

public class HelloListBug {

    public static void main(String[] args) throws InterruptedException {
        var nbThreads = 4;
        var threads = new Thread[nbThreads];
        var list = new ArrayList<Integer>();

        IntStream.range(0, nbThreads).forEach(j -> {
            Runnable runnable = () -> {
                for (var i = 0; i < 5000; i++) {
                    list.add(i);
                }
            };

            threads[j] = new Thread(runnable);
            threads[j].start();
        });

        for (var thread : threads) {
            thread.join();
        }

        System.out.println("le programme est fini");
        System.out.println("Taille obtenue : "+list.size());
    }
}
