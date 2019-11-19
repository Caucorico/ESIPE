package exercice01;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class Main {

    public final static int THREAD_NUMBER = 4;

    private Thread[] threads = new Thread[THREAD_NUMBER];

    private void stopThreadById(int id) {
        threads[id].interrupt();
    }

    private void readStdIn() throws IOException {
        System.out.println("enter a thread id:");
        try (var input = new InputStreamReader(System.in);
             var reader = new BufferedReader(input)) {
            String line;
            while ((line = reader.readLine()) != null) {
                var threadId = Integer.parseInt(line);
                this.stopThreadById(threadId);
            }
        }
    }

    private void addThread(int id, Thread thread) {
        threads[id] = thread;
    }

    public static void main(String[] args) {
        Main main = new Main();

        Runnable runnable = () -> {
            while ( !Thread.interrupted() ) {
                try {
                    Thread.sleep(1000);
                } catch ( InterruptedException e ) {
                    Thread.currentThread().interrupt();
                    continue;
                }
                System.out.println("Thread n°" + Thread.currentThread().getName());
            }

            /* Not thread safe for the display */
            System.out.println(Thread.currentThread().getName() + " stopped");
        };

        for ( var i = 0 ; i < THREAD_NUMBER ; i++ ) {
            var thread = new Thread(runnable);
            thread.setName(i+"");
            main.addThread(i, thread);
            thread.start();
        }

        while ( true ) {
            try {
                main.readStdIn();
            } catch (IOException e) {
                continue;
            }
        }
    }
}
