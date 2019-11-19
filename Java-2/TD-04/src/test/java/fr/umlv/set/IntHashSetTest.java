package fr.umlv.set;

import static org.junit.jupiter.api.Assertions.*;

import java.time.Duration;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.stream.IntStream;

import org.junit.jupiter.api.*;

@SuppressWarnings("static-method")
public class IntHashSetTest {

  //Q3
  @Test
  public void shouldAddAString() {
    var set = new IntHashSet();
    set.add(1);
  }
  
  @Test
  public void shouldAddAnInteger() {
    var set = new IntHashSet();
    set.add(31_133);
  }
  
  @Test
  public void shouldAddWithoutErrors() {
    var set = new IntHashSet();
    IntStream.range(0, 100).map(i -> i * 2 + 1).forEach(set::add);
  }
  
  @Test
  public void shouldNotTakeTooLongToAddTheSameNumberMultipleTimes() {
    var set = new IntHashSet();
    assertTimeout(Duration.ofMillis(5_000), () -> IntStream.range(0, 1_000_000).map(i -> 42).forEach(set::add));
  }
  
  @Test
  public void shouldAnswerZeroWhenAskingForSizeOfEmptySet() {
    var set = new IntHashSet();
    assertEquals(0, set.size());
  }
  
  @Test
  public void shouldNotAddTwiceTheSameAndComputeSizeAccordingly() {
    var set = new IntHashSet();
    set.add(3);
    assertEquals(1, set.size());
    set.add(-777);
    assertEquals(2, set.size());
    set.add(3);
    assertEquals(2, set.size());
    set.add(-777);
    assertEquals(2, set.size());
  }

  //Q5
  @Test
  public void shouldNotUseNullAsAParameterForForEach() {
    var set = new IntHashSet();
    set.add(3);
    assertThrows(NullPointerException.class, () -> set.forEach(null));
  }

  @Test
  public void shouldDoNoThingWhenForEachCalledOnEmptySet() {
    var set = new IntHashSet();
    set.forEach(__ -> fail("should not be called"));
  }
  
  @Test
  public void shouldCompteTheSumOfAllTheElementsInASetUsingForEachAngGetTheSameAsTheFormula() {
    var length = 100;
    var set = new IntHashSet();
    IntStream.range(0, length).forEach(set::add);
    var sum = new int[] { 0 };
    set.forEach(value -> sum[0] += value);
    assertEquals(length * (length - 1) / 2, sum[0]);
  }
  
  @Test
  public void shouldComputeIndenticalSetAndHashSetUsingForEachAndHaveSameSize() {
    var set = new IntHashSet();
    IntStream.range(0, 100).forEach(set::add);
    var hashSet = new HashSet<Integer>();
    set.forEach(hashSet::add);
    assertEquals(set.size(), hashSet.size());
  }
  
  @Test
  public void shouldAddAllTheElementsOfASetToAListUsingForEach() {
    var set = new IntHashSet();
    IntStream.range(0, 100).forEach(set::add);
    var list = new ArrayList<Integer>();
    set.forEach(list::add);
    list.sort(null);
    IntStream.range(0, 100).forEach(i -> assertEquals(i, (int)list.get(i)));
  }
  
  //Q6
  @Test
  public void shouldNotFindAnythingContainedInAnEmptySet() {
    var set = new IntHashSet();
    assertFalse(set.contains(4));
    assertFalse(set.contains(7));
    assertFalse(set.contains(1));
    assertFalse(set.contains(0));
  }
  
  @Test
  public void shouldNotFindAnIntegerBeforeAddingItButShouldFindItAfter() {
    var set = new IntHashSet();
    for(int i = 0; i < 10; i++) {
      assertFalse(set.contains(i));
      set.add(i);
      assertTrue(set.contains(i));
    }
  }
  
  @Test
  public void shoulAddAndTestContainsForAnExtremalValue() {
    var set = new IntHashSet();
    assertFalse(set.contains(Integer.MIN_VALUE));
    set.add(Integer.MIN_VALUE);
    assertTrue(set.contains(Integer.MIN_VALUE));
    set.add(Integer.MAX_VALUE);
    assertTrue(set.contains(Integer.MAX_VALUE));
  }
}
