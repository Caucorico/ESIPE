package exercice03;

import java.util.Random;

public class FindPrime {
  public static boolean isPrime(long l) {
    if (l <= 1)
      return false;
    for (long i = 2L; i <= l / 2; i++) {
      if (l % i == 0) {
        return false;
      }
    }
    return true;
  }

  public static void main(String[] args) throws InterruptedException {
    var nbThreads = 4;
    var rdv = new StupidRendezVous<Long>();
    for (var i = 0; i < nbThreads; i++) {
      var fi = i;
      var thread = new Thread(() -> {
        var random = new Random();
        for (;;) {
          var nb = 10_000_000_000L + (random.nextLong() % 10_000_000_000L);
          if (isPrime(nb)) {
            rdv.set(nb);
            System.out.println("A prime number was found in thread " + fi);
            return;
          }
        }
      });
      thread.setDaemon(true);
      thread.start();
    }
    Long prime = rdv.get();
    System.out.println("I found a large prime number : " + prime);
  }
}
