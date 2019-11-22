package exercice01;

import request.Answer;
import request.Request;

import java.util.Optional;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.stream.Collectors;

public class Cheapest {
    private final ArrayBlockingQueue<Answer> responsesList;

    private final String item;
    private final int timeoutMilliPerRequest;

    public Cheapest(String item, int timeoutMilliPerRequest) {
        this.item = item;
        this.timeoutMilliPerRequest = timeoutMilliPerRequest;
        this.responsesList = new ArrayBlockingQueue<>(Request.ALL_SITES.size());
    }

    /**
     * @return the cheapest price for item if it is sold
     */
    public Optional<Answer> retrieve() throws InterruptedException {
        var i = 0;
        Answer cheapest = null;

        Request.ALL_SITES.forEach( e -> {
            Runnable runnable = () -> {
                var request = new Request(e, item);
                try {
                    responsesList.put(request.request(timeoutMilliPerRequest));
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
    }

    public static void main(String[] args) throws InterruptedException {
        System.out.println(new Cheapest("tortank",2_000).retrieve());
    }
}
