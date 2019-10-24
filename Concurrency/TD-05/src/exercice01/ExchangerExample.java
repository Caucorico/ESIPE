package exercice01;

//import java.util.concurrent.Exchanger; We use local.

public class ExchangerExample {
  public static void main(String[] args) throws InterruptedException {
    var exchanger = new Exchanger<String>();
    new Thread(() -> {
      try {
        System.out.println("thread 1 " + exchanger.exchange("foo1"));
      } catch (InterruptedException e) {
        throw new AssertionError(e);
      }
    }).start();
    System.out.println("main " + exchanger.exchange(null));
  }
}
