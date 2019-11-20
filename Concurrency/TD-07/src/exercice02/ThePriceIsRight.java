package exercice02;

import java.util.ArrayList;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;

public class ThePriceIsRight {

    private final int price;
    private final int participantNumber;

    private final ReentrantLock winnerLock = new ReentrantLock(false);

    private final ReentrantLock gameFinishedLock = new ReentrantLock(false);
    private final Condition gameFinishedStopCondition = gameFinishedLock.newCondition();

    private final ReentrantLock propositionsLock = new ReentrantLock(false);

    private final ArrayList<Map.Entry<Thread, Integer>> propositions;

    private boolean gameFinished;

    private Optional<Thread> winner;

    public ThePriceIsRight(int price, int participantNumber) {

        if ( price < 0 ) throw new IllegalArgumentException("price cannot be negative");
        if ( participantNumber < 1 ) throw new IllegalArgumentException("There is at least one participant");

        this.price = price;
        this.participantNumber = participantNumber;
        this.propositions = new ArrayList<>(participantNumber);

        this.winnerLock.lock();
        try {
            this.winner = Optional.empty();
        } finally {
            this.winnerLock.unlock();
        }
    }

    private boolean isFinished() {
        this.gameFinishedLock.lock();
        try {
            return this.gameFinished;
        } finally {
            this.gameFinishedLock.unlock();
        }
    }

    private boolean updateFinished() {
        this.propositionsLock.lock();
        try {
            this.gameFinishedLock.lock();
            try {
                this.gameFinished = this.propositions.size() == this.participantNumber;
                if ( this.gameFinished ) {
                    this.gameFinishedStopCondition.signalAll();
                }
                return this.gameFinished;
            } finally {
                this.gameFinishedLock.unlock();
            }
        } finally {
            this.propositionsLock.unlock();
        }
    }

    private Optional<Thread> computeWinner() {
        this.propositionsLock.lock();
        try {
            this.winnerLock.lock();
            try {
                while ( !this.isFinished() ) {
                    try {
                        this.gameFinishedStopCondition.await();
                    } catch (InterruptedException e) {
                        return Optional.empty();
                    }
                }
                Thread winner = null;
                var smallerGap = Integer.MAX_VALUE;
                /* TODO : stop that loop when interrupted */
                for ( var i = 0 ; i < this.participantNumber ; i++ ) {
                    var currentTuple = this.propositions.get(i);
                    var gap = this.price - currentTuple.getValue();
                    if ( Math.abs(gap) < smallerGap ) {
                        winner = currentTuple.getKey();
                        smallerGap = Math.abs(gap);
                    }
                }
                if ( winner == null ) throw new IllegalThreadStateException("N participant found !");
                return Optional.of(winner);
            } finally {
                this.winnerLock.unlock();
            }
        } finally {
            this.propositionsLock.unlock();
        }
    }

    private Optional<Thread> getWinner() {
        this.propositionsLock.lock();
        try {
            this.winnerLock.lock();
            try {
                if ( this.winner.isEmpty() ){
                    return this.computeWinner();
                }
                return this.winner;
            } finally {
                this.winnerLock.unlock();
            }
        } finally {
            this.propositionsLock.unlock();
        }
    }

    public boolean propose(int proposition){
        this.propositionsLock.lock();
        try {
            if ( this.isFinished() ) return false;
            this.propositions.add(Map.entry(Thread.currentThread(), proposition));
            this.updateFinished();
        } finally {
            this.propositionsLock.unlock();
        }

        this.gameFinishedLock.lock();
        try {
            while ( !this.gameFinished && !Thread.interrupted() ) {
                try {
                    this.gameFinishedStopCondition.await();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }

            if ( !this.gameFinished )
            {
                this.propositions.removeIf(e -> e.getKey() == Thread.currentThread());
                this.gameFinished = true;
                this.gameFinishedStopCondition.signalAll();
                return false;
            }

            var winner = this.getWinner();
            if ( winner.isEmpty() ) return false;
            if ( winner.get().equals(Thread.currentThread()) ) return true;
            return false;
        } finally {
            this.gameFinishedLock.unlock();
        }
    }

}
