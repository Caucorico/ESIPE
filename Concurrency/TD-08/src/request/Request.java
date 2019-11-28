package request;

import java.util.Collections;
import java.util.List;
import java.util.Set;

public class Request {

    private final String site;
    private final String item;
    private final Object lock = new Object();
    private boolean started;

    private final static String[] ARRAY_ALL_SITES = { "amazon.fr", "amazon.uk", "darty.fr", "fnac.fr", "boulanger.fr",
            "cdiscount.fr", "tombeducamion.fr", "leboncoin.fr", "ebay.fr", "ebay.com", "laredoute.fr",
            "les3suisses.fr" };
    private final static Set<String> SET_ALL_SITES = Set.of(ARRAY_ALL_SITES);
    public final static List<String> ALL_SITES = Collections.unmodifiableList(List.of(ARRAY_ALL_SITES));

    public Request(String site, String item) {
        if (!SET_ALL_SITES.contains(site))
            throw new IllegalStateException();
        this.site = site;
        this.item = item;
    }

    @Override
    public String toString() {
        return item + "@" + site;
    }

    /**
     * Performs the request the price for the item on the site waiting at most
     * timeoutMilli milliseconds. The returned Answered is not guaranteed to be
     * successful.
     *
     * This method can only be called once. All further calls will throw an
     * IllegalStateException
     *
     *
     * @param timeoutMilli
     * @throws InterruptedException
     */
    public Answer request(int timeoutMilli) throws InterruptedException {
        synchronized (lock) {
            if (started)
                throw new IllegalStateException();
            started = true;
        }
        System.out.println("DEBUG : starting request for " + item + " on " + site);
        if (item.equals("pokeball")) {
            System.out.println("DEBUG : " + item + " is not available on " + site);
            return new Answer(site, item, null);
        }
        long hash1 = Math.abs((site + "|" + item).hashCode());
        long hash2 = Math.abs((item + "|" + site).hashCode());
        if ((hash1 % 1000 < 400) || ((hash1 % 1000) * 2 > timeoutMilli)) { // simulating timeout
            Thread.sleep(timeoutMilli);
            System.out.println("DEBUG : Request " + toString() + " has timeout");
            return new Answer(site, item, null);
        }
        Thread.sleep((hash1 % 1000) * 2);
        if ((hash1 % 1000 < 500)) {
            System.out.println("DEBUG : " + item + " is not available on " + site);
            return new Answer(site, item, null);
        }
        int price = (int) (hash2 % 1000) + 1;
        System.out.println("DEBUG : " + item + " costs " + price + " on " + site);
        return new Answer(site, item, price);
    }

    public static void main(String[] args) throws InterruptedException {
        Request request = new Request("amazon.fr", "pikachu");
        Answer answer = request.request(5_000);
        if (answer.isSuccessful()) {
            System.out.println("The price is " + answer.getPrice());
        } else {
            System.out.println("The price could not be retrieved from the site");
        }
    }
}
