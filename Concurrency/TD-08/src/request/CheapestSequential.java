package request;

import java.util.Comparator;
import java.util.Optional;

public class CheapestSequential {

    private final String item;
    private final int timeoutMilliPerRequest;

    public CheapestSequential(String item, int timeoutMilliPerRequest) {
        this.item = item;
        this.timeoutMilliPerRequest = timeoutMilliPerRequest;
    }

    /**
     * @return the cheapest price for item if it is sold
     */
    public Optional<Answer> retrieve() throws InterruptedException {
        return (Optional<Answer>) Request.ALL_SITES.stream()
            .map(e -> {
                var request = new Request(e, this.item);
                try {
                    var answer = request.request(timeoutMilliPerRequest);
                    if ( answer.isSuccessful() ) return Optional.of(answer);
                    return Optional.empty();
                } catch (InterruptedException ex) {
                    return Optional.empty();
                }
            })
            .filter(Optional::isPresent)
            .map( e -> e.get() )
            .min(Comparator.comparingInt(a -> ((Answer) a).getPrice()));
    }

    public static void main(String[] args) throws InterruptedException {
        var agregator = new CheapestSequential("pikachu", 2_000);
        var answer = agregator.retrieve();
        System.out.println(answer); // Optional[pikachu@darty.fr : 214]
    }
}
