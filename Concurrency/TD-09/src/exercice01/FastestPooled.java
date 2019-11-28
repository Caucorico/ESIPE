package exercice01;

import request.Answer;
import request.Request;

import java.util.ArrayList;
import java.util.Optional;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;

public class FastestPooled {
    private final String item;
    private final int timeoutMilliPerRequest;
    private final int poolSize;

    public FastestPooled(String item, int timeoutMilliPerRequest, int poolSize) {
        this.item = item;
        this.timeoutMilliPerRequest = timeoutMilliPerRequest;
        this.poolSize = poolSize;
    }

    public Optional<Answer> retrieve() throws InterruptedException, ExecutionException {
        var callables = new ArrayList<Callable<Answer>>();
        Request.ALL_SITES.forEach(site -> {
            callables.add(() -> {
                var request = new Request(site, item);
                return request.request(timeoutMilliPerRequest);
            });
        });

        var executorService = Executors.newFixedThreadPool(poolSize);
        var result = executorService.invokeAny(callables);
        if ( result.isSuccessful() ) return Optional.of(result);
        return Optional.empty();
    }

    public static void main(String[] args) throws InterruptedException, ExecutionException {
        System.out.println(new FastestPooled("tortank",2_000, 10).retrieve());
    }
}
