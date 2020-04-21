package fr.umlv.structconc;

public class Vectorized {
  public static int sumLoop(int[] array) {
    var sum = 0;
    for(var value: array) {
      sum += value;
    }
    return sum;
  }
}
