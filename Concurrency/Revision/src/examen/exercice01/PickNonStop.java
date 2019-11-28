package examen.exercice01;

import java.util.*;
import java.util.Map.Entry;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class PickNonStop {
	private final Object cardsLock = new Object();
	private final ArrayDeque<Integer> cards = new ArrayDeque<>();
	private final Object playersLock = new Object();
	private final HashMap<Thread, Integer> players = new HashMap<Thread, Integer>();

	private final static int MAX_CARD = 9;

	public void startNewGame(int nbCards) {
		Random random = new Random();
		synchronized (cardsLock) {
			random.ints(nbCards, -MAX_CARD, MAX_CARD + 1).forEach(cards::offer);
		}
	}

	public Integer pick() {
		var player = Thread.currentThread();
		Integer card;
		synchronized (cardsLock) {
			card = cards.poll();
		}

		//System.out.println("player " + player.getName() + " picks " + card);
		if (card == null){ // there is no card left
			return null;
		}
		synchronized ( playersLock ) {
			players.merge(player, card, Integer::sum);
		}
		return card;
	}

	public Optional<String> winner() {
		synchronized ( playersLock ) {
			return players.entrySet().stream().max(Entry.comparingByValue())
					.map(e -> e.getKey().getName() + " wins with " + e.getValue());
		}
	}

	public static void main(String[] args) {
		var pick = new PickNonStop();
		pick.startNewGame(100);
		var nbThread = 10;
		List<Thread> threads = IntStream.range(0, nbThread).mapToObj(i -> {
			var thread = new Thread(() -> {
				pick.pick();
				pick.pick();
			});
			thread.setName("Thread " + i);
			thread.start();
			return thread;
		}).collect(Collectors.toList());

		threads.forEach(t -> {
			try {
				t.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		});
	}
}
