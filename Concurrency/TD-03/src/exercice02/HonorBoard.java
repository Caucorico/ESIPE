package exercice02;

public class HonorBoard {
  private String firstName;
  private String lastName;
  private final Object lock = new Object();
  
  public void set(String firstName, String lastName) {
    synchronized ( lock ) {
      this.firstName = firstName;
      this.lastName = lastName;
    }
  }
  
  @Override
  public String toString() {
    synchronized ( lock ) {
      return firstName + ' ' + lastName;
    }
  }
  
  public static void main(String[] args) {
    HonorBoard board = new HonorBoard();
    new Thread(() -> {
      for(;;) {
        board.set("John", "Doe");
      }
    }).start();
    
    new Thread(() -> {
      for(;;) {
        board.set("Jane", "Odd");
      }
    }).start();
    
    new Thread(() -> {
      for(;;) {
        System.out.println(board);
      }
    }).start();
  }
}
