package exercice03;

import exercice01.Cheapest;
import request.Answer;
import request.Request;

import java.util.Optional;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.stream.IntStream;

public class CheapestPooledWithFixedThreadPool {

    private final String item;
    private final FixedThreadPool pool;
    private final int timeoutMilliPerRequest;

    public CheapestPooledWithFixedThreadPool(String item, int timeoutMilliPerRequest, int pooleSize) {
        this.item = item;
        this.pool = new FixedThreadPool(pooleSize);
        this.timeoutMilliPerRequest = timeoutMilliPerRequest;
    }

    public Optional<Answer> retrieve() throws InterruptedException {
        var sitesQueue = new ArrayBlockingQueue<String>(Request.ALL_SITES.size(), false, Request.ALL_SITES);
        var answersQueue = new ArrayBlockingQueue<Answer>(Request.ALL_SITES.size());
        Answer cheapest = null;
        var i = 0;

        Task t = () -> {
            var request = new Request(sitesQueue.take(), item);
            answersQueue.put(request.request(timeoutMilliPerRequest));
        };
        IntStream.range(0, Request.ALL_SITES.size())
            .forEach( in -> pool.submit(t));
        pool.start();

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

        pool.stop();

        return Optional.ofNullable(cheapest);
    }

    public static void main(String[] args) throws InterruptedException {
        System.out.println(new Cheapest("tortank",2_000).retrieve());
    }
}
