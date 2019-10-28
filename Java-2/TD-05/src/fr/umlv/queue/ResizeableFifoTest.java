package fr.umlv.queue;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.NoSuchElementException;

import org.junit.jupiter.api.Test;

@SuppressWarnings("static-method")
public class ResizeableFifoTest {
  @Test
  public void shouldResizeWhenAddingMoreThanCapacityElements() {
    var fifo = new ResizeableFifo<String>(1);
    fifo.offer("foo");
    fifo.offer("bar");
    assertEquals(2, fifo.size());
    assertEquals("foo", fifo.poll());
    assertEquals("bar", fifo.poll());
    assertEquals(0, fifo.size());
  }

  @Test
  public void shouldKeepElementsInOrderWhenResizing() {
    var fifo = new ResizeableFifo<String>(2);
    fifo.offer("foo");
    fifo.poll();
    fifo.offer("bar");
    fifo.offer("baz");
    fifo.offer("bat");
    assertEquals(3, fifo.size());
    assertEquals("bar", fifo.poll());
    assertEquals("baz", fifo.poll());
    assertEquals("bat", fifo.poll());
    assertEquals(0, fifo.size());
  }

  @Test
  public void shouldResizeALot() {
    var fifo = new ResizeableFifo<Integer>(1);
    fifo.offer(-1);
    fifo.poll();
    for (int i = 0; i < 10_000; i++) {
      fifo.offer(i);
    }
    for (int i = 0; i < 10_000; i++) {
      assertEquals(i, (int) fifo.poll());
    }
  }

  
  // --- ResizeableFifo has a different semantics than Fifo
  
  @Test
  public void shouldGetAnErrorWhenPollingFromEmptyFifo() {  
    var fifo = new ResizeableFifo<>(1);
    //assertThrows(IllegalStateException.class, () -> fifo.poll());
    assertNull(fifo.poll());
  }

  @Test
  public void shouldGetAnErrorWhenOfferingToFullFifo() {
    var fifo = new ResizeableFifo<Integer>(1);
    fifo.offer(43);
    //assertThrows(IllegalStateException.class, () -> fifo.offer(7));
    fifo.offer(7);
  }
  
  
  // --- Fifo tests 
  
  @Test
  public void shouldGetAnErrorWhenCapacityIsNonPositive() {
    assertThrows(IllegalArgumentException.class, () -> new ResizeableFifo<>(-3));
  }

  @Test
  public void shouldGetAnErrorWhenCapacityIsZero() {
    assertThrows(IllegalArgumentException.class, () -> new ResizeableFifo<>(0));
  }

  

  @Test
  public void shouldGetAnErrorWhenOfferingNull() {
    var fifo = new ResizeableFifo<>(234);
    assertThrows(NullPointerException.class, () -> fifo.offer(null));
  }

  @Test
  public void shouldGetOfferedValueWhenPolling() {
    var fifo = new ResizeableFifo<Integer>(2);
    fifo.offer(9);
    assertEquals(9, (int) fifo.poll());
    fifo.offer(2);
    fifo.offer(37);
    assertEquals(2, (int) fifo.poll());
    fifo.offer(12);
    assertEquals(37, (int) fifo.poll());
    assertEquals(12, (int) fifo.poll());
  }

  @Test
  public void shouldGetOfferedValueWhenPollingWithMixedTypes() {
    var fifo = new ResizeableFifo<>(20);
    for (var i = 0; i < 20; i++) {
      fifo.offer(i);
    }
    assertEquals(0, (int) fifo.poll());
    fifo.offer("foo");
    for (var i = 1; i < 20; i++) {
      assertEquals(i, (int) fifo.poll());
    }
    assertEquals("foo", fifo.poll());
  }

  @Test
  public void shoulgGetACorrectSize() {
    var fifo = new ResizeableFifo<String>(2);
    assertEquals(0, fifo.size());
    fifo.offer("foo");
    assertEquals(1, fifo.size());
    fifo.offer("bar");
    assertEquals(2, fifo.size());
    fifo.poll();
    assertEquals(1, fifo.size());
    fifo.poll();
    assertEquals(0, fifo.size());
  }

  @Test
  public void shouldAnswerZeroWhenAskedForTheSizeOfAnEmptyFifo() {
    var fifo = new ResizeableFifo<>(1);
    assertEquals(0, fifo.size());
  }

  @Test
  public void shouldAnswerOneWhenAskedForTheSizeAfterOneOffer() {
    var fifo = new ResizeableFifo<String>(1);
    fifo.offer("dooh");
    assertEquals(1, fifo.size());
  }

  @Test
  public void shouldFindFifoEmptyOnlyAfterRemovingAllElement() {
    var fifo = new ResizeableFifo<String>(2);
    assertTrue(fifo.isEmpty());
    fifo.offer("oof");
    assertFalse(fifo.isEmpty());
    fifo.offer("rab");
    assertFalse(fifo.isEmpty());
    fifo.poll();
    fifo.poll();
    assertTrue(fifo.isEmpty());
  }

  @Test
  public void shouldPrintEmptyFifo() {
    var fifo = new ResizeableFifo<>(23);
    assertEquals("[]", fifo.toString());
  }

  @Test
  public void shouldPrintResizeableFifoWithOneElement() {
    var fifo = new ResizeableFifo<String>(23);
    fifo.offer("joe");
    assertEquals("[joe]", fifo.toString());
  }

  @Test
  public void shouldPrintFifoWithTwoElements() {
    var fifo = new ResizeableFifo<Integer>(23);
    fifo.offer(1456);
    fifo.offer(8390);
    assertEquals("[1456, 8390]", fifo.toString());
  }

  @Test
  public void shouldBeAbleToAddMoreThanCapacityAfterRemoval() {
    var fifo = new ResizeableFifo<String>(2);
    fifo.offer("foo");
    fifo.poll();
    fifo.offer("1");
    fifo.offer("2");
    assertEquals("[1, 2]", fifo.toString());
  }

  @Test
  public void shouldNotAffectFifoWhenPrinting() {
    var fifo = new ResizeableFifo<Integer>(200);
    var list = new ArrayList<Integer>();
    for (var i = 0; i < 100; i++) {
      fifo.offer(i);
      list.add(i);
    }
    assertEquals(list.toString(), fifo.toString());
    for (var i = 0; i < 100; i++) {
      assertEquals(i, (int) fifo.poll());
    }
  }

  @Test
  public void shouldPrintFifoInTheSameWayAsAList() {
    var fifo = new ResizeableFifo<Integer>(99);
    var list = new ArrayList<Integer>();
    for (var i = 0; i < 99; i++) {
      fifo.offer(i);
      list.add(i);
    }
    assertEquals(list.toString(), fifo.toString());
  }

  @Test
  public void shouldGetTheRightTypeOfIterator() {
    var fifo = new ResizeableFifo<String>(1);
    Iterator<String> it = fifo.iterator();
    assertNotNull(it);
  }

  @Test
  public void shouldGetAnErrorWhenAskingNextWhenDoesNotHaveNext() {
    var fifo = new ResizeableFifo<String>(1);
    fifo.offer("bar");
    fifo.poll();
    var it = fifo.iterator();
    assertThrows(NoSuchElementException.class, () -> it.next());
  }

  @Test
  public void shouldNotGetSideEffectsWhenUsingIteratorHasNext() {
    var fifo = new ResizeableFifo<Integer>(3);
    fifo.offer(117);
    fifo.offer(440);
    var it = fifo.iterator();
    assertTrue(it.hasNext());
    assertTrue(it.hasNext());
    assertEquals(117, (int) it.next());
    assertTrue(it.hasNext());
    assertTrue(it.hasNext());
    assertEquals(440, (int) it.next());
    assertFalse(it.hasNext());
    assertFalse(it.hasNext());
  }

  @Test
  public void shouldIterateProperlyWhenTheNumberofOffersOvertakesOriginalCapacity() {
    var fifo = new ResizeableFifo<Integer>(2);
    fifo.offer(42);
    fifo.poll();
    fifo.offer(55);
    fifo.offer(333);
    var it = fifo.iterator();
    assertTrue(it.hasNext());
    assertTrue(it.hasNext());
    assertEquals(55, (int) it.next());
    assertTrue(it.hasNext());
    assertTrue(it.hasNext());
    assertEquals(333, (int) it.next());
    assertFalse(it.hasNext());
    assertFalse(it.hasNext());
  }

  @Test
  public void shouldBeAbleToIterateTwice() {
    var fifo = new ResizeableFifo<Integer>(1);
    fifo.offer(898);

    var it = fifo.iterator();
    assertTrue(it.hasNext());
    assertTrue(it.hasNext());
    assertEquals(898, (int) it.next());
    assertFalse(it.hasNext());
    assertFalse(it.hasNext());
    var it2 = fifo.iterator();
    assertTrue(it2.hasNext());
    assertTrue(it2.hasNext());
    assertEquals(898, (int) it2.next());
    assertFalse(it2.hasNext());
    assertFalse(it2.hasNext());
  }

  @Test
  public void shouldGetConsistentAnswersFromHasNextWhenEmpty() {
    var fifo = new ResizeableFifo<>(1);
    var it = fifo.iterator();
    assertFalse(it.hasNext());
    assertFalse(it.hasNext());
    assertFalse(it.hasNext());
  }

  @Test
  public void shouldGetConsistentAnswersFromHasNextWhenNotEmpty() {
    var fifo = new ResizeableFifo<>(1);
    fifo.offer("one");
    var it = fifo.iterator();
    assertTrue(it.hasNext());
    assertTrue(it.hasNext());
    assertTrue(it.hasNext());
  }

  @Test
  public void shouldIterateOverALargeNumberOfElements() {
    var fifo = new ResizeableFifo<Integer>(10_000);
    for (int i = 0; i < 10_000; i++) {
      fifo.offer(i);
    }
    var i = 0;
    var it = fifo.iterator();
    while (it.hasNext()) {
      assertEquals(i++, (int) it.next());
    }
    assertEquals(10_000, fifo.size());
  }

  @Test
  public void shouldGetAnErrorWhenTryingToUseIteratorRemove() {
    var fifo = new ResizeableFifo<String>(1);
    fifo.offer("foooo");
    assertThrows(UnsupportedOperationException.class, () -> fifo.iterator().remove());
  }

  //this test needs a lot of memory (more than 8 gigs)
  // so it is disabled by default
  // use the option -Xmx9g when running the VM
  /* @Test
  public void shouldNotGetAnOverflowErrorWhenIteratingOverAnAlmostMaximalCapacityFifo() {
    var fifo = new ResizeableFifo<Integer>(Integer.MAX_VALUE - 8);
    for(var i = 0; i < Integer.MAX_VALUE / 2; i++) {
      fifo.offer(i % 100);
      fifo.poll();
    }
    for(var i = 0; i < Integer.MAX_VALUE - 8; i++) {
      fifo.offer(i % 100);
    }
    var counter = 0;
    for(var it = fifo.iterator(); it.hasNext(); counter = (counter + 1) % 100) {
      assertEquals(counter, (int)it.next());
    }
  }*/
  
  @Test
  public void shouldBeAbleToUseImplicitForEachLoop() {
    var fifo = new ResizeableFifo<Integer>(100);
    fifo.offer(222);
    fifo.poll();

    for (var i = 0; i < 100; i++) {
      fifo.offer(i);
    }
    var i = 0;
    for (int value : fifo) {
      assertEquals(i++, value);
    }
    assertEquals(100, fifo.size());
  }

  // ---

  @Test
  public void shoulGetNullWhenPeekingFromAnEmptyFifo() {
    var fifo = new ResizeableFifo<String>(1);
    assertNull(fifo.peek());
  }

  @Test
  public void shouldPeekCorrectlyFromNonEmptyFifo() {
    var fifo = new ResizeableFifo<Integer>(1);
    fifo.offer(117);
    assertEquals(117, (int) fifo.peek());
    fifo.poll();
    fifo.offer(145);
    fifo.offer(541);
    assertEquals(145, (int) fifo.peek());
    fifo.poll();
    assertEquals(541, (int) fifo.peek());
  }

  @Test
  public void shouldGetAnErrorWhenTryingToGetElementFromEmptyFifo() {
    var fifo = new ResizeableFifo<Integer>(1);
    assertThrows(NoSuchElementException.class, fifo::element);
  }

  @Test
  public void shouldAddAllElementsFromOneFifoToAnother() {
    var fifo = new ResizeableFifo<Integer>(1);
    for (int i = 0; i < 3; i++) {
      fifo.offer(i);
    }

    var fifo2 = new ResizeableFifo<>(1);
    fifo2.addAll(fifo);

    assertEquals(fifo2.size(), fifo.size());

    var it2 = fifo2.iterator();
    var it = fifo.iterator();
    while (it2.hasNext()) {
      assertTrue(it.hasNext());
      assertEquals(it2.next(), it.next());
    }
  }
}
