package fr.umlv.set;

import static org.junit.jupiter.api.Assertions.*;

import java.time.Duration;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;

@SuppressWarnings("static-method")
public class DynamicHashSetTest {
  @Test
  public void shouldAddAString() {
    var set = new DynamicHashSet<String>();
    set.add("hello");
  }
  
  @Test
  public void shouldAddAnInteger() {
    var set = new DynamicHashSet<Integer>();
    set.add(31_133);
  }
  
  @Test
  public void shouldAddWithoutErrors() {
    var set = new DynamicHashSet<Integer>();
    IntStream.range(0, 100).map(i -> i * 2 + 1).forEach(set::add);
  }
    
  @Test
  public void shouldNotTakeTooLongToAddDifferentElementsMultipleTimes() {
    var set = new DynamicHashSet<Integer>();
    assertTimeout(Duration.ofMillis(3_000), () -> IntStream.range(0, 1_000_000).map(i -> i * 2).forEach(set::add));
  }
  
  @Test
  public void shouldNotTakeTooLongToAddTheSameNumberMultipleTimes() {
    var set = new DynamicHashSet<Integer>();
    assertTimeout(Duration.ofMillis(3_000), () -> {
      for(var i = 0; i < 1_000_000; i++) {
        set.add(i);
      }  
    });
  }
  
  @Test
  public void shouldGetAnErrorWhenAddingNull() {
    var set = new DynamicHashSet<>();
    assertThrows(NullPointerException.class, () -> set.add(null));
  }
  
  @Test
  public void shouldAnswerZeroWhenAskingForSizeOfEmptySet() {
    var set = new DynamicHashSet<>();
    assertEquals(0, set.size());
  }
  
  @Test
  public void shouldNotAddTwiceTheSameAndComputeSizeAccordingly() {
    var set = new DynamicHashSet<Integer>();
    set.add(3);
    assertEquals(1, set.size());
    set.add(-777);
    assertEquals(2, set.size());
    set.add(3);
    assertEquals(2, set.size());
    set.add(-777);
    assertEquals(2, set.size());
  }

  @Test
  public void shouldNotUseNullAsAParameterForForEach() {
    var set = new DynamicHashSet<Integer>();
    set.add(3);
    assertThrows(NullPointerException.class, () -> set.forEach(null));
  }

  @Test
  public void shouldDoNoThingWhenForEachCalledOnEmptySet() {
    var set = new DynamicHashSet<>();
    set.forEach(__ -> fail("should not be called"));
  }
  
  @Test
  public void shouldCompteTheSumOfAllTheElementsInASetUsingForEachAngGetTheSameAsTheFormula() {
    var length = 100;
    var set = new DynamicHashSet<Integer>();
    IntStream.range(0, length).forEach(set::add);
    var sum = new int[] { 0 };
    set.forEach(value -> sum[0] += value);
    assertEquals(length * (length - 1) / 2, sum[0]);
  }
  
  @Test
  public void shouldComputeIndenticalSetAndHashSetUsingForEachAndHaveSameSize() {
    var set = new DynamicHashSet<Integer>();
    IntStream.range(0, 100).forEach(set::add);
    var hashSet = new HashSet<Integer>();
    set.forEach(hashSet::add);
    assertEquals(set.size(), hashSet.size());
  }
  
  @Test
  public void shouldAddAllTheElementsOfASetToAListUsingForEach() {
    var set = new DynamicHashSet<Integer>();
    IntStream.range(0, 100).forEach(set::add);
    var list = new ArrayList<Integer>();
    set.forEach(list::add);
    list.sort(null);
    IntStream.range(0, 100).forEach(i -> assertEquals(i, (int)list.get(i)));
  }
  
  @Test
  public void shouldTakeTheRightTypeOfConsumerInForEach() {
    var set = new DynamicHashSet<String>();
    set.add("foo");
    Consumer<Object> consumer = o -> assertTrue(o.equals("foo")); 
    set.forEach(consumer);
  }
  
  @Test
  public void shouldNotFindAnythingInAnEmptySet() {
    var set = new DynamicHashSet<String>();
    assertFalse(set.contains("baz"));
  }
  
  @Test
  public void shouldNotFindAnythingContainedInAnEmptySet() {
    var set = new DynamicHashSet<Integer>();
    assertFalse(set.contains(4));
    assertFalse(set.contains(7));
    assertFalse(set.contains(1));
    assertFalse(set.contains(0));
  }
  
  @Test
  public void shouldNotFindAnIntegerBeforeAddingItButShouldFindItAfter() {
    var set = new DynamicHashSet<Integer>();
    for(var i = 0; i < 100; i++) {
      assertFalse(set.contains(i));
      set.add(i);
      assertTrue(set.contains(i));
    }
  }
  
  @Test
  public void shoulAddAndTestContainsForAnExtremalValue() {
    var set = new DynamicHashSet<Integer>();
    assertFalse(set.contains(Integer.MIN_VALUE));
    set.add(Integer.MIN_VALUE);
    assertTrue(set.contains(Integer.MIN_VALUE));
    set.add(Integer.MIN_VALUE);
    assertTrue(set.contains(Integer.MIN_VALUE));
  }
  
  @Test
  public void shouldGetAnErrorWhenSearchingForNull() {
    var set = new DynamicHashSet<>();
    assertThrows(NullPointerException.class, () -> set.contains(null));
  }

  
  @Test
  public void shouldNotTakeTooLongToAskContains() {
  var set = new DynamicHashSet<Integer>();
    IntStream.range(0, 100_000).forEach(set::add);
    assertTimeout(Duration.ofMillis(5_000), () -> IntStream.range(0, 100_000).forEach(set::contains));
  }
  
  @Test
  public void shouldAddAllTheElementsOfAList() {
    var set = new DynamicHashSet<String>();
    set.addAll(List.of("hello", "boy"));
    assertEquals(2, set.size());
    set.forEach(s -> assertTrue(s.equals("hello") || s.equals("boy")));
  }
  
  @Test
  public void shouldAddAllTheElementsOfASet() {
    var set = new DynamicHashSet<String>();
    set.add("bar");
    set.addAll(List.of("bar"));
    assertEquals(1, set.size());
    set.forEach(s -> assertEquals("bar", s));
  }
  
  @Test
  public void shouldTakeTheRightTypeOfCollectionAsArgumentOfAddAll() {
    var set = new DynamicHashSet<>();
    var hashSet = new HashSet<String>();
    set.addAll(hashSet);
  }
}
