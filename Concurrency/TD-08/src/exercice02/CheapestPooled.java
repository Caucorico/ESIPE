package exercice02;

import exercice01.Cheapest;
import request.Answer;
import request.Request;

import java.util.Optional;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class CheapestPooled {

    /*
     * Quel type de BlockingQueue peut-on utiliser pour sitesQueue et answersQueue.
     * Nous pouvont utiliser des ArrayBlockingQueue :
     */
    private final ArrayBlockingQueue<String> sitesQueue = new ArrayBlockingQueue<String>(Request.ALL_SITES.size(), false, Request.ALL_SITES);
    private final ArrayBlockingQueue<Answer> answersQueue = new ArrayBlockingQueue<>(Request.ALL_SITES.size());

    private final String item;
    private final int timeoutMilliPerRequest;
    private final int pooleSize;

    public CheapestPooled(String item, int timeoutMilliPerRequest, int pooleSize) {
        this.item = item;
        this.timeoutMilliPerRequest = timeoutMilliPerRequest;
        this.pooleSize = pooleSize;
    }

    public Optional<Answer> retrieve() throws InterruptedException {
        var i = 0;
        Answer cheapest = null;

        Stream<Thread> threads = IntStream.range(0, pooleSize).mapToObj(
                e -> {
                    Runnable runnable = () -> {
                        Request request = null;
                        while ( Thread.interrupted() ) {
                            try {
                                request = new Request(sitesQueue.take(), item);
                            } catch (InterruptedException ex) {
                                Thread.currentThread().interrupt();
                                continue;
                            }
                            try {
                                answersQueue.put(request.request(timeoutMilliPerRequest));
                            } catch (InterruptedException ex){
                                Thread.currentThread().interrupt();
                                continue;
                            }
                        }
                    };
                    var thread = new Thread(runnable);
                    thread.start();
                    return thread;
                }
        );

        while ( i < Request.ALL_SITES.size() ) {
            var response = answersQueue.take();
            if ( response.isSuccessful() )
            {
                if ( cheapest == null || response.getPrice() < cheapest.getPrice() ) {
                    cheapest = response;
                }
            }
            i++;
        }

        threads.forEach(Thread::interrupt);

        return Optional.ofNullable(cheapest);
    }

    public static void main(String[] args) throws InterruptedException {
        System.out.println(new Cheapest("tortank",2_000).retrieve());
    }
}
