package exercice02;

import exercice01.Cheapest;
import request.Answer;
import request.Request;

import java.util.Optional;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.stream.IntStream;

public class CheapestPooled {

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

    /*public Optional<Answer> retrieve() throws InterruptedException {
        var i = 0;
        Answer cheapest = null;

        IntStream.range(0, pooleSize).forEach(
                e -> {
                    Runnable runnable = () -> {
                        if ( )
                        var request = new Request(e, item);
                        try {
                            answersQueue.put(request.request(timeoutMilliPerRequest));
                        } catch (InterruptedException ex){
                            return;
                        }
                    };
                    var thread = new Thread(runnable);
                    thread.start();
                }
        );

        sitesQueue.forEach( e -> {
            Runnable runnable = () -> {
                var request = new Request(e, item);
                try {
                    answersQueue.put(request.request(timeoutMilliPerRequest));
                } catch (InterruptedException ex){
                    return;
                }
            };
            var thread = new Thread(runnable);
            thread.start();
        });

        while ( i < Request.ALL_SITES.size() ) {
            var response = responsesList.take();
            if ( response.isSuccessful() )
            {
                if ( cheapest == null || response.getPrice() < cheapest.getPrice() ) {
                    cheapest = response;
                }
            }
            i++;
        }

        return Optional.ofNullable(cheapest);
    }*/
}
