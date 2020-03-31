package fr.umlv.conc;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;

public class SpinLock {

  /*
   * Réponses aux questions :
   *
   * 1) Un lock réentrant est un lock que l'on peut reprendre sur le même Thread récursivement.
   *
   * 2) Le code suivant ne peut pas être Thread-safe.En effet, même si counter était valatile, l'opération ++ n'est pas
   *    atomique.
   *
   * 3) a) Si le lock est déjà pris, le thread qui essaye de le prendre doit attendre que le jeton soit de nouveau disponible.
   *    b) Le problème, c'est qu'on ne peut pas dire au Thread d'attendre x temps, (sinon attente active), il faut donc attendre autrement.
   *    c) La méthode onSpinWait permet au Thread d'attendtre intélligemant. Il n'essayera pas de reprendre le lock à chaque cycle.
   */

  private volatile boolean lock = false;

  private static final VarHandle HANDLE;

  static {
    try {
      HANDLE = MethodHandles.lookup().findVarHandle(SpinLock.class, "lock", boolean.class);
    } catch (NoSuchFieldException | IllegalAccessException e) {
      throw new AssertionError(e);
    }
  }

  public void lock() {
    while (!HANDLE.compareAndSet(this, false, true)) {
      Thread.onSpinWait();
    }
  }
  
  public void unlock() {
    lock = false;
  }

  public boolean tryLock() {
    return HANDLE.compareAndSet(this, false, true);
  }
  
  public static void main(String[] args) throws InterruptedException {
    var runnable = new Runnable() {
      private int counter;
      private final SpinLock spinLock = new SpinLock();
      
      @Override
      public void run() {
        for(int i = 0; i < 1_000_000; i++) {
          if ( !spinLock.tryLock() ) {
            i--;
            continue;
          }
          try {
            counter++;
          } finally {
            spinLock.unlock();
          }
        }
      }
    };
    var t1 = new Thread(runnable);
    var t2 = new Thread(runnable);
    t1.start();
    t2.start();
    t1.join();
    t2.join();
    System.out.println("counter " + runnable.counter);
  }
}
