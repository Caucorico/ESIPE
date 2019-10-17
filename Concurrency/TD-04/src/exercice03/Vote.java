package exercice03;

import java.util.HashMap;
import java.util.concurrent.ConcurrentHashMap;

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
        }

        return "";
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
                        System.out.println("The winner is " + vote.vote("0"));
                    } catch (InterruptedException e) {
                        throw new AssertionError(e);
                    }
                })
                .start();
        System.out.println("The winner is " + vote.vote("0"));
    }
}
