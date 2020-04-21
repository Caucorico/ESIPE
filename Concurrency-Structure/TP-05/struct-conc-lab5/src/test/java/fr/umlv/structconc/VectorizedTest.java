package fr.umlv.structconc;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

@SuppressWarnings("static-method")
public class VectorizedTest {
  private static Stream<Arguments> provideIntArrays() {
    return IntStream.of(0, 1, 10, 100, 1000, 10_000, 100_000)
        .mapToObj(i -> new Random(0).ints(i, 0, 1000).toArray())
        .map(array -> Arguments.of(array, Arrays.stream(array).reduce(0, Integer::sum)));
  }

  @ParameterizedTest
  @MethodSource("provideIntArrays")
  public void sum(int[] array, int expected) {
    assertEquals(expected, Vectorized.sumLoop(array));
  }
}