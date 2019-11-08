package fr.umlv.bag;

import static java.util.stream.Collectors.toList;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeout;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.sql.Timestamp;
import java.time.Duration;
import java.util.*;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;

@SuppressWarnings("static-method")
public class BagTest {
  // Q2
  
  @Test
  public void shouldCreateASimpleBag() {
    Bag.createSimpleBag();
  }
  
  @Test
  public void shouldAddSeveralReturnTheRightCount() {
    var bag = Bag.<String>createSimpleBag();
    assertEquals(10, bag.add("splash", 10));
  }

  @Test
  public void shouldAddOneByOneAndReturnTheRightCount() {
    Bag<String> bag = Bag.createSimpleBag();
    assertEquals(1, bag.add("foo", 1));
    assertEquals(1, bag.add("bar", 1));
    assertEquals(2, bag.add("foo", 1));
  }

  @Test
  public void shouldAddIdenticalElementsAndReturnTheRightCount() {
    var bag = Bag.<String>createSimpleBag();
    assertEquals(2, bag.add("blob", 2));
    assertEquals(3, bag.add("blob", 1));
  }

  @Test
  public void shouldGetAnErrorWhenAddingNull() {
    var bag = Bag.createSimpleBag();
    assertThrows(NullPointerException.class, () -> bag.add(null, 1));
  }
  @Test
  public void shouldGetAnErrorWhenAddingZero() {
    var bag = Bag.createSimpleBag();
    assertThrows(IllegalArgumentException.class, () -> bag.add("foo", 0));
  }
  @Test
  public void shouldGetAnErrorWhenAddingWithNegativeCount() {
    var bag = Bag.createSimpleBag();
    assertThrows(IllegalArgumentException.class, () -> bag.add("foo", -1));
  }

  @Test
  public void shouldGetAnErrorWhenAddingWithNegativeCountEvenIfTheTotalIsPositive() {
    var bag = Bag.createSimpleBag();
    bag.add("buzz", 1);
    assertThrows(IllegalArgumentException.class, () -> bag.add("buzz", -1));
  }

  @Test
  public void shouldCountCorrecltyWheneverPresentOrNot() {
    var bag = Bag.<Integer>createSimpleBag();
    bag.add(1, 1);
    bag.add(1, 1);
    bag.add(42, 1);
    assertEquals(2, bag.count(1));
    assertEquals(1, bag.count(42));
    assertEquals(0, bag.count(153));
    assertEquals(0, bag.count("foo"));
  }

  @Test
  public void shouldGetAnErrorWhenCountingNull() {
    var bag = Bag.createSimpleBag();
    assertThrows(NullPointerException.class, () -> bag.count(null));
  }

  @Test
  public void shouldAddInBatchesAndReturnTheRightCount() {
    var bag = Bag.createSimpleBag();
    bag.add(1, 3);
    assertEquals(3, bag.count(1));
    bag.add("foobar", 7);
    assertEquals(7, bag.count("foobar"));
  }

  
  // Q3
  
  @Test
  public void shouldIterateWithForEachOverAnElementWithMultiplesOccurences() {
    var bag = Bag.<Integer>createSimpleBag();
    bag.add(117, 3);
    bag.forEach(e -> assertEquals(117, (int) e));
  }

  @Test
  public void shouldIterateWithForEachOverDifferentElements() {
    var bag = Bag.<Integer>createSimpleBag();
    bag.add(34, 1);
    bag.add(48, 1);
    var set = new HashSet<Integer>();
    bag.forEach(set::add);
    assertEquals(Set.of(34, 48), set);
  }

  @Test
  public void shouldNotIterateWithForEachOverAnEmptyBag() {
    Bag<Object> empty = Bag.createSimpleBag();
    empty.forEach(__ -> fail("should not be called"));
  }

  @Test
  public void shouldGetAnErrorWithForEachAndANullConsumer() {
    var bag = Bag.createSimpleBag();
    assertThrows(NullPointerException.class, () -> bag.forEach(null));
  }

  
  // Q4 and Q5
  
  @Test
  public void shouldAnIteratorProperlyTyped() {
    var bag = Bag.<String>createSimpleBag();
    Iterator<String> it = bag.iterator();
    assertNotNull(it);
  }
  
  @Test
  public void shouldIterateProperlyWithIteratorAndFailWhenAskedForNonExistantNext() {
    var bag = Bag.<String>createSimpleBag();
    bag.add("hello", 2);
    var iterator = bag.iterator();
    assertEquals("hello", iterator.next());
    assertEquals("hello", iterator.next());
    assertThrows(NoSuchElementException.class, () -> iterator.next());
  }

  @Test
  public void shoulBeAbleToAskHasNextAsManyTimesAsYouWishWithoutMessingUpWithNext() {
    var bag = Bag.<String>createSimpleBag();
    bag.add("bob", 1);
    var iterator = bag.iterator();
    for (int i = 0; i < 100; i++) {
      assertEquals(true, iterator.hasNext());
    }
    assertEquals("bob", iterator.next());
    assertFalse(iterator.hasNext());
  }

  @Test
  public void shoulBeAbleToGetAndUseTwoDifferentIterators() {
    var bag = Bag.<String>createSimpleBag();
    bag.add("bang", 1);

    var iterator1 = bag.iterator();
    assertTrue(iterator1.hasNext());
    assertEquals("bang", iterator1.next());
    assertFalse(iterator1.hasNext());

    var iterator2 = bag.iterator();
    assertTrue(iterator2.hasNext());
    assertEquals("bang", iterator2.next());
    assertFalse(iterator2.hasNext());
  }

  @Test
  public void shouldHaveNoNextWithIteratorOverAnEmptyBag() {
    assertFalse(Bag.createSimpleBag().iterator().hasNext());
  }

  @Test
  public void shouldGetAnErrorWhenAskingNextOnEmptyIterator() {
    var iterator = Bag.createSimpleBag().iterator();
    assertThrows(NoSuchElementException.class, iterator::next);
  }

  @Test
  public void shouldGetAnErrorWhenTryingToRemoveWithoutCallingNextBefore() {
    var iterator = Bag.createSimpleBag().iterator();
    assertThrows(UnsupportedOperationException.class, iterator::remove);
  }

  @Test
  public void shouldIterateWithIteratorOverAFewElementsWithMultiplesOccurences() {
    var bag = Bag.<Integer>createSimpleBag();
    var set = new HashSet<Integer>();
    for (int i = 0; i < 1000; i++) {
      bag.add(i % 3, 1);
      set.add(i % 3);
    }
    var set2 = new HashSet<Integer>();
    var iterator = bag.iterator();
    assertTrue(iterator.hasNext());
    for (; iterator.hasNext();) {
      set2.add(iterator.next());
    }
    assertEquals(set, set2);
    assertFalse(iterator.hasNext());
  }

  @Test
  public void shouldIterateWithIteratorOverManyElementsWithoutDisturbingHasNext() {
    var bag = Bag.<String>createSimpleBag();
    for (int i = 0; i < 1_000; i++) {
      bag.add(Integer.toString(i), 1);
    }
    var iterator = bag.iterator();
    assertTrue(iterator.hasNext());
    for (int i = 0; i < 1_000; i++) {
      if (i % 11 == 0) {
        for (int j = 0; j < 7; j++) {
          iterator.hasNext();
        }
      }
      iterator.next();
    }
    assertFalse(iterator.hasNext());
  }

  
  // Q6
  
  @Test
  public void shouldBeAbleToDoAnEnhancedLoopOverABag() {
    var bag = Bag.<Integer>createSimpleBag();
    var list = IntStream.range(0, 100).boxed().collect(toList());
    list.forEach(element -> bag.add(element, 1));
    
    var list2 = new ArrayList<Integer>();
    for(var element: bag) {
      list2.add(element);
    }
    assertEquals(list, list2);
  }
  
  
  // Q7 and Q8
  
  @Test
  public void shouldGetACollectionShouldHaveTheRightType() {
    var bag = Bag.<String>createSimpleBag();
    Collection<String> collection = bag.asCollection();
    assertNotNull(collection);
  }
  
  @Test
  public void shouldGetACollectionWithExactlyOneElement() {
    var bag = Bag.<Integer>createSimpleBag();
    bag.add(4, 1);
    var collection = bag.asCollection();
    assertEquals(1, collection.size());
    assertFalse(collection.isEmpty());
    assertTrue(collection.contains(4));
    assertFalse(collection.contains("hello"));
  }
  
  @Test
  public void shouldGetACollectionThatReflectModificationsAfterCreation() {
    var bag = Bag.<String>createSimpleBag();
    var collection = bag.asCollection();
    assertTrue(collection.isEmpty());
    bag.add("hello", 2);
    assertEquals(2, collection.size());
    assertFalse(collection.isEmpty());
    assertTrue(collection.contains("hello"));
    assertFalse(collection.contains(42));
  }

  @Test
  public void shouldNotBeAbleToModifyTheCollectionReturnedByAsCollection() {
    var collection = Bag.createSimpleBag().asCollection();
    assertThrows(UnsupportedOperationException.class, () -> collection.add("hello"));
  }

  @Test
  public void shouldGetTheSameElementsTheRightNumberOfTimesWithAsCollection() {
    var bag = Bag.<Integer>createSimpleBag();
    bag.add(4, 1);
    bag.add(7, 2);
    bag.add(17, 5);
    bag.add(15, 3);
    var collection = bag.asCollection();
    assertEquals(Set.of(4, 7, 17, 15), new HashSet<>(collection));
    assertEquals(11, collection.size());
  }

  @Test
  public void shouldGetAccessToTheCollectionAsAViewWithAsCollection() {
    var bag = Bag.<Integer>createSimpleBag();
    var collection = bag.asCollection();
    bag.add(69, 2);
    assertEquals(69, (int) collection.iterator().next());
    assertEquals(69, (int) collection.iterator().next());
  }

  @Test
  public void shouldNotRecreateTheWholeBagWithAsCollection() {
    var bag = Bag.<Integer>createSimpleBag();
    IntStream.range(0, 100_000).forEach(i -> bag.add(i, 1 + i));
    
    assertTimeout(Duration.ofMillis(1), () -> {
      assertEquals(100_001 * 100_000 / 2, bag.asCollection().size());  
    });
  }

  @Test
  public void shouldBeEffcientWhenUsingContainsFromAsCollection() {
    var bag = Bag.<Integer>createSimpleBag();
    IntStream.range(0, 100_000).forEach(i -> bag.add(i, 1 + i));

    assertTimeout(Duration.ofMillis(1_000), () -> {
      for (int i = 1; i < 100_000; i++) {
        assertFalse(bag.asCollection().contains(-i));
      }
    });
  }

  
  // Q9
  
  @Test
  public void shouldGetTheElementsInInsertionOrderWithCreateOrderedByInsertionBag() {
    var bag = Bag.<Integer>createOrderedByInsertionBag();
    bag.add(4, 1);
    bag.add(7, 2);
    bag.add(17, 1);
    bag.add(15, 1);
    assertEquals(List.of(4, 7, 7, 17, 15), new ArrayList<>(bag.asCollection()));
  }

  @Test
  public void shouldGetTheElementsInInsertionOrderWhenUsingIterator() {
    var bag = Bag.<Integer>createOrderedByInsertionBag();
    var list = new ArrayList<Integer>();
    for (int i = 0; i < 1000; i++) {
      bag.add(i, 1);
      list.add(i);
    }
    var list2 = new ArrayList<Integer>();
    bag.iterator().forEachRemaining(list2::add);
    assertEquals(list, list2);
  }

  @Test
  public void shouldGetTheElementsInComparatorOrderWhenUsingIterator() {
    Bag<Integer> bag = Bag.createOrderedByInsertionBag();
    var list = new ArrayList<Integer>();
    for (int j = 0; j < 5; j++) {
      for (int i = 0; i < 100; i++) {
        bag.add(i, 1);
      }
    }

    for (int i = 0; i < 100; i++) {
      for (int j = 0; j < 5; j++) {
        list.add(i);
      }
    }

    var list2 = new ArrayList<Integer>();
    bag.iterator().forEachRemaining(list2::add);
    assertEquals(list, list2);
  }

  
  // Q10
  
  @Test
  public void shouldGetTheElementsInComparatorOrderWhenUsingForEach() {
    var bag = Bag.createOrderedByElementBag(Integer::compareTo);
    bag.add(31, 1);
    bag.add(16, 3);

    ArrayList<Integer> list = new ArrayList<>();
    bag.forEach(list::add);

    assertEquals(List.of(16, 16, 16, 31), list);
  }
  
  @Test
  public void shouldGetTheElementsInComparatorOrderWithCreateOrderedByElementBag() {
    var bag = Bag.createOrderedByElementBag(Integer::compare);
    bag.add(4, 1);
    bag.add(7, 2);
    bag.add(17, 1);
    bag.add(15, 1);
    assertEquals(List.of(4, 7, 7, 15, 17), new ArrayList<>(bag.asCollection()));
  }
  
  @Test
  public void shouldBeAbleoCreateAnOrderedBagWithAComparatorDefinedOnASuperType() {
    Bag<String> bag = Bag.<String>createOrderedByElementBag((Object o1, Object o2) -> o1.toString().compareTo(o2.toString()));
    bag.add("tomato", 2);
    bag.add("zoo", 1);
    bag.add("elephant", 1);
    assertEquals(List.of("elephant", "tomato", "tomato", "zoo"), new ArrayList<>(bag.asCollection()));
  }
  
  @Test
  public void shouldGetMultipleElementsInComparatorOrderWhenUsingIterator() {
    var bag = Bag.createOrderedByElementBag(String::compareTo);
    bag.add("hello", 3);
    bag.add("boy", 2);
    bag.add("girl", 1);

    var list = new ArrayList<String>();
    bag.iterator().forEachRemaining(list::add);

    assertEquals(List.of("boy", "boy", "girl", "hello", "hello", "hello"), list);
  }

  
  @Test
  public void shouldTheBagCreatedFromACollectionTypedCorrectly() {
    Bag<String> bag = Bag.createOrderedByElementBagFromCollection(List.<String>of());
    assertNotNull(bag);
  }
  
  @Test
  public void shouldGetTheElementsInComparableOrderWithCreateOrderedByElementBagFromCollection() {
    var bag = Bag.createOrderedByElementBagFromCollection(List.of("foo", "zoom", "bar", "zoom"));
    var list = new ArrayList<String>();
    bag.iterator().forEachRemaining(list::add);

    assertEquals(List.of("bar", "foo", "zoom", "zoom"), list);
  }

  @Test
  public void shouldGetTheElementsInComparableOrderWithCreateOrderedByElementBagFromCollection2() {
    var bag = Bag.createOrderedByElementBagFromCollection(List.of(new Timestamp(7), new Timestamp(3)));
    var list2 = new ArrayList<Timestamp>();
    bag.iterator().forEachRemaining(list2::add);

    assertEquals(List.of(new Timestamp(3), new Timestamp(7)), list2);
  }
}