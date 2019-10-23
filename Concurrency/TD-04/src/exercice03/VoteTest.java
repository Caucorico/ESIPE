package exercice03;

import org.junit.jupiter.api.Test;

import java.time.Duration;

import static org.junit.jupiter.api.Assertions.*;

public class VoteTest {

    @Test
    public void SimpleVote() throws Exception {
        Vote vote = new Vote(3);
        new Thread(
                () -> {
                    try {
                        Thread.sleep(1_000);
                        assertEquals("0", vote.vote("1"));
                    } catch (InterruptedException e) {
                        throw new AssertionError(e);
                    }
                })
                .start();
        new Thread(
                () -> {
                    try {
                        Thread.sleep(500);
                        assertEquals("0", vote.vote("0"));
                    } catch (InterruptedException e) {
                        throw new AssertionError(e);
                    }
                })
                .start();
        assertEquals("0", vote.vote("0"));
    }

    @Test
    public void VoteAlone() throws InterruptedException {
        Vote vote = new Vote(1);
        assertEquals("0", vote.vote("0"));
    }




    @Test
    public void VoteWithATie() throws Exception {
        Vote vote = new Vote(2);
        new Thread(
                () -> {
                    try {
                        Thread.sleep(1_000);
                        assertEquals("0", vote.vote("1"));
                    } catch (InterruptedException e) {
                        throw new AssertionError(e);
                    }
                })
                .start();
        assertEquals("0", vote.vote("0"));
    }

    @Test
    public void ManyVotes() throws Exception {
        Vote vote = new Vote(2);
        for(int i=0;i<4;i++) {
            new Thread(
                    () -> {
                        try {
                            Thread.sleep(1_000);
                            assertEquals("0", vote.vote("1"));
                        } catch (InterruptedException e) {
                            throw new AssertionError(e);
                        }
                    })
                    .start();
        }
        assertEquals("0", vote.vote("0"));
    }
}
