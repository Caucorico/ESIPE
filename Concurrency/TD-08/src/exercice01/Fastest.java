package exercice01;

import request.Answer;
import request.Request;

import java.util.Comparator;
import java.util.Optional;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.stream.Collectors;

public class Fastest {

    private final ArrayBlockingQueue<Answer> responsesList;

    private final String item;
    private final int timeoutMilliPerRequest;

    public Fastest(String item, int timeoutMilliPerRequest) {
        this.item = item;
        this.timeoutMilliPerRequest = timeoutMilliPerRequest;
        this.responsesList = new ArrayBlockingQueue<>(Request.ALL_SITES.size());
    }

    /**
     * @return the cheapest price for item if it is sold
     */
    public Optional<Answer> retrieve() throws InterruptedException {
        var i = 0;

        var threads = Request.ALL_SITES.stream()
            .map( e -> {
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
                return thread;
            })
            .collect(Collectors.toList());

        while ( i < Request.ALL_SITES.size() ) {
            var response = responsesList.take();
            if ( response.isSuccessful() ) return Optional.of(response);
            i++;
        }

        threads.forEach(Thread::interrupt);

        return Optional.empty();
    }

    public static void main(String[] args) throws InterruptedException {
        var agregator = new Fastest("tortank", 2_000);
        var answer = agregator.retrieve();
        System.out.println("RESPONSE" + answer);
    }
}
