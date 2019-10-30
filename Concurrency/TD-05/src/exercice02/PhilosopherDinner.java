package exercice02;

import java.util.Arrays;
import java.util.stream.IntStream;

public class PhilosopherDinner {
  private final boolean[] taken;
  private final Object[] takenLock;

  public PhilosopherDinner(int forkCount) {
    boolean[] taken = new boolean[forkCount];
    Object[] takenLock = new Object[forkCount];
    Arrays.fill(taken, false);
    Arrays.setAll(takenLock, i -> new Object());
    this.taken = taken;
    this.takenLock = takenLock;
  }

  public void eat(int index) throws InterruptedException {
    var fork1 = takenLock[index];
    var fork2 = takenLock[(index + 1) % takenLock.length];
    synchronized ( fork1 ) {
      while ( taken[(index + 1) % taken.length]) takenLock[index].wait();
      taken[index] = true;
      synchronized ( fork2 ) {
        /* Marche seulement si le wait de fork1 dans fork2 fait aussi wait fork2 (J'en suis pas sur) */
        /* Dans la doc, il est écrit que wait met en pause le thread et non pas le lock */
        while ( taken[(index + 1) % taken.length]) takenLock[index].wait();
        taken[(index + 1) % taken.length] = true;
        System.out.println("philosopher " + index + " eat");
        taken[(index + 1) % taken.length] = false;
        /* Cette ligne a aussi l'air de notifier les fork1, ce que je ne comprends pas... */
        /* Mais ça, la doc ne le dit pas... Le wait précédent aurait-il appelé la wait fork2 également ?*/
        takenLock[(index + 1) % takenLock.length].notifyAll();
      }
      taken[index] = false;
    }

    /*
    Conclusion de cette fonction : Si j'avais plus de temps, j'aurais creuser plus.
    Pour moi, avec mes connaissances actuelles, cette fonction ne marche pas.
    */
  }

  public static void main(String[] args) {
    var dinner = new PhilosopherDinner(5);
    IntStream.range(0, 5).forEach(i -> {
      new Thread(() -> {
        for (;;) {
          try {
            dinner.eat(i);
          } catch (InterruptedException e) {
            throw new RuntimeException(e);
          }
        }
      }).start();
    });
  }
}
