package exercice01;

import request.Answer;
import request.Request;

import java.util.ArrayList;
import java.util.Optional;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.stream.IntStream;

public class CheapestPooled {

    private final String item;
    private final int timeoutMilliPerRequest;
    private final int poolSize;

    public CheapestPooled(String item, int timeoutMilliPerRequest, int poolSize) {
        this.item = item;
        this.timeoutMilliPerRequest = timeoutMilliPerRequest;
        this.poolSize = poolSize;
    }

    public Optional<Answer> retrieve() throws InterruptedException {
        var callables = new ArrayList<Callable<Answer>>();
        Request.ALL_SITES.forEach(site -> {
            callables.add(() -> {
                var request = new Request(site, item);
                return request.request(timeoutMilliPerRequest);
            });
        });

        var executorService = Executors.newFixedThreadPool(poolSize);
        var result = executorService.invokeAll(callables);

        return result.stream()
            .map( future -> {
                try {
                    return future.get();
                } catch (InterruptedException | ExecutionException e) {
                    throw new AssertionError(e);
                }
            })
            .filter(Answer::isSuccessful)
            .min(Answer.ANSWER_COMPARATOR);
    }

    public static void main(String[] args) throws InterruptedException {
        System.out.println(new CheapestPooled("tortank",2_000, 10).retrieve());
    }
}
