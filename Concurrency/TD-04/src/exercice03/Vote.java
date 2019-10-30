package exercice03;

import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

public class Vote {
    private final Object lock = new Object();
    private final int maxVoteNumber;
    private int voteNumber = 0;
    private final HashMap<String, Integer> hashMap = new HashMap<>();

    public Vote(int maxVoteNumber) {
        this.maxVoteNumber = maxVoteNumber;
    }

    private void addCandidature(String candidate) {
        synchronized (lock) {
            hashMap.put(candidate, hashMap.getOrDefault(candidate, 0)+1);
            voteNumber++;
            lock.notifyAll();
        }
    }

    public String vote(String candidate) throws InterruptedException {
        synchronized (lock) {
            if ( voteNumber < maxVoteNumber ) {
                addCandidature(candidate);
            }
            while (voteNumber < maxVoteNumber) {
                lock.wait();
            }
            return getWinner();
        }
    }

    public String getWinner() {
        synchronized (lock) {
            var max = hashMap.entrySet().stream().max(Comparator.comparing(Map.Entry::getValue));
            if ( max.isEmpty() ) throw new IllegalStateException("Winner not found !");
            else return max.get().getKey();
        }
    }

    public static void main(String[] args) throws InterruptedException {
        Vote vote = new Vote(3);
        new Thread(
                () -> {
                    try {
                        Thread.sleep(10_000);
                        System.out.println("The winner is " + vote.vote("1"));
                    } catch (InterruptedException e) {
                        throw new AssertionError(e);
                    }
                })
                .start();
        new Thread(
                () -> {
                    try {
                        Thread.sleep(5_000);
                        System.out.println("The winner is " + vote.vote("1"));
                    } catch (InterruptedException e) {
                        throw new AssertionError(e);
                    }
                })
                .start();
        System.out.println("The winner is " + vote.vote("0"));
    }
}
