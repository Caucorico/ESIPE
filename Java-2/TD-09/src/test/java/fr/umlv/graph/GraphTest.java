package fr.umlv.graph;

import static java.util.stream.Collectors.toSet;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeout;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.lang.reflect.Modifier;
import java.time.Duration;
import java.util.HashSet;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

@SuppressWarnings("static-method")
public class GraphTest {
  interface GraphFactory {
    <T> Graph<T> createGraph(int vertexCount);
  }
  static Stream<GraphFactory> graphFactoryProvider() {
    return Stream.of(Graph::createMatrixGraph/*, Graph::createNodeMapGraph*/);
  }
  
  
  // Q3
  
  @ParameterizedTest @Tag("Q3")
  @MethodSource("graphFactoryProvider")
  public void testCreateGraph(GraphFactory factory) {
    var graph = factory.createGraph(50);
  }
  
  @ParameterizedTest @Tag("Q3")
  @MethodSource("graphFactoryProvider")
  public void testCreateGraphSignature(GraphFactory factory) {
    Graph<String> graph = factory.<String>createGraph(50);
  }
  
  @ParameterizedTest @Tag("Q3")
  @MethodSource("graphFactoryProvider")
  public void testInvalidNodeCount(GraphFactory factory) {
    assertThrows(IllegalArgumentException.class, () -> factory.createGraph(-17));
  }
  
  @ParameterizedTest @Tag("Q3")
  @MethodSource("graphFactoryProvider")
  public void implementationHidden(GraphFactory factory) {
    var graph = factory.createGraph(10);
    assertFalse(Modifier.isPublic(graph.getClass().getModifiers()));
  }

  
  // Q4
  
  @ParameterizedTest @Tag("Q4")
  @MethodSource("graphFactoryProvider")
  public void testGetWeightEmpty(GraphFactory factory) {
    var nodeCount = 20;
    Graph<Object> graph = factory.createGraph(nodeCount);
    for (var i = 0; i < nodeCount; i++) {
      for (var j = 0; j < nodeCount; j++) {
        assertTrue(graph.getWeight(i, j).isEmpty());
      }
    }
  }

  @ParameterizedTest @Tag("Q4")
  @MethodSource("graphFactoryProvider")
  public void testHasEdgeValid(GraphFactory factory) {
    var graph = factory.createGraph(5);
    assertAll(
        () -> assertThrows(IndexOutOfBoundsException.class, () -> graph.getWeight(-1, 3)),
        () -> assertThrows(IndexOutOfBoundsException.class, () -> graph.getWeight(2, -1)),
        () -> assertThrows(IndexOutOfBoundsException.class, () -> graph.getWeight(5, 2)),
        () -> assertThrows(IndexOutOfBoundsException.class, () -> graph.getWeight(3, 5))
        );
  }

  
  // Q5
  
  @ParameterizedTest @Tag("Q5")
  @MethodSource("graphFactoryProvider")
  public void testAddEdge(GraphFactory factory) {
    var graph = factory.<Integer>createGraph(7);
    graph.addEdge(3, 4, 2);
    assertEquals(2, (int) graph.getWeight(3, 4).orElseThrow());
  }
  
  @ParameterizedTest @Tag("Q5")
  @MethodSource("graphFactoryProvider")
  public void testAddEdgeWithAString(GraphFactory factory) {
    var graph = factory.<String>createGraph(10);
    graph.addEdge(7, 8, "hello");
    assertEquals("hello", graph.getWeight(7, 8).orElseThrow());
    assertFalse(graph.getWeight(4, 3).isPresent());
  }
  
  @ParameterizedTest @Tag("Q5")
  @MethodSource("graphFactoryProvider")
  public void testAddEdgeNullWeight(GraphFactory factory) {
    var graph = factory.<Integer>createGraph(7);
    assertThrows(NullPointerException.class, () -> graph.addEdge(3, 4, null));
  }

  @ParameterizedTest @Tag("Q5")
  @MethodSource("graphFactoryProvider")
  public void testAddEdgeTwice(GraphFactory factory) {
    var graph = factory.<String>createGraph(7);
    graph.addEdge(3, 4, "foo");
    graph.addEdge(3, 4, "bar");
    assertEquals("bar", graph.getWeight(3, 4).orElseThrow());
  }

  @ParameterizedTest @Tag("Q5")
  @MethodSource("graphFactoryProvider")
  public void testAddEdgeValid(GraphFactory factory) {
    var graph = factory.createGraph(5);
    assertAll(
        () -> assertThrows(IndexOutOfBoundsException.class, () -> graph.addEdge(-1, 3, 7)),
        () -> assertThrows(IndexOutOfBoundsException.class, () -> graph.addEdge(2, -1, 8)),
        () -> assertThrows(IndexOutOfBoundsException.class, () -> graph.addEdge(5, 2, 9)),
        () -> assertThrows(IndexOutOfBoundsException.class, () -> graph.addEdge(3, 5, 10)));
  }

  @ParameterizedTest @Tag("Q5")
  @MethodSource("graphFactoryProvider")
  public void testAddEdgeALot(GraphFactory factory) {
    var graph = factory.createGraph(17);
    var random = ThreadLocalRandom.current();
    IntStream.range(0, 1000).forEach(index -> {
      var i = random.nextInt(17);
      var j = random.nextInt(17);
      var value = random.nextInt(10_000) - 5_000;
      graph.addEdge(i, j, value);
      assertEquals(value, (int) graph.getWeight(i, j).orElseThrow());
    });
  }

  
  // Q6

  @ParameterizedTest @Tag("Q6")
  @MethodSource("graphFactoryProvider")
  public void testEdgesOneEdge(GraphFactory factory) {
    var graph = factory.createGraph(3);
    graph.addEdge(1, 0, 2);
    graph.edges(0, (src, dst, weight) -> {
      assertEquals(1, src);
      assertEquals(0, dst);
      assertEquals(2, (int) weight);
    });
  }
  
  @ParameterizedTest @Tag("Q6")
  @MethodSource("graphFactoryProvider")
  public void testEdgesOnEdge(GraphFactory factory) {
    var graph = factory.<String>createGraph(13);
    graph.addEdge(3, 7, "foo");
    graph.edges(0, (src, dst, weight) -> {
      assertEquals(3, src);
      assertEquals(7, dst);
      assertEquals("foo", weight);
    });
  }
  
  @ParameterizedTest @Tag("Q6")
  @MethodSource("graphFactoryProvider")
  public void testEdgesNoEdge(GraphFactory factory) {
    var graph = factory.createGraph(17);
    graph.edges(0, (src, dst, weight) -> fail("should not be called"));
  }

  @ParameterizedTest @Tag("Q6")
  @MethodSource("graphFactoryProvider")
  public void testEdgesNullConsumer(GraphFactory factory) {
    assertThrows(NullPointerException.class, () -> factory.createGraph(17).edges(0, null));
  }


  @ParameterizedTest @Tag("Q6")
  @MethodSource("graphFactoryProvider")
  public void testEdgesALot(GraphFactory factory) {
    var nodeCount = 200;
    var graph = factory.createGraph(nodeCount);
    for (var i = 0; i < nodeCount; i++) {
      for (var j = 0; j < nodeCount; j++) {
        graph.addEdge(i, j, i % 10 + j);
      }
    }
    assertTimeout(Duration.ofMillis(2_000), () -> {
      for (var i = 0; i < nodeCount; i++) {
        graph.edges(i, (src, dst, weight) -> {
          assertEquals(src % 10 + dst, (int) weight);
        });
      }  
    });
  }

  
  // Q7
  
  @ParameterizedTest @Tag("Q7")
  @MethodSource("graphFactoryProvider")
  public void testNeighborsSimple(GraphFactory factory) {
    var graph = factory.<Integer>createGraph(6);
    graph.addEdge(1, 2, 222);
    graph.addEdge(1, 5, 555);
    
    var iterator = graph.neighborIterator(1);
    assertEquals(2, (int)iterator.next());
    assertEquals(5, (int)iterator.next());
  }
  
  @ParameterizedTest @Tag("Q7")
  @MethodSource("graphFactoryProvider")
  public void testNeighborsEmptyHasNext(GraphFactory factory) {
    var graph = factory.createGraph(6);
    var iterator = graph.neighborIterator(0);
    assertFalse(iterator.hasNext());
    assertFalse(iterator.hasNext());
    assertFalse(iterator.hasNext());
  }

  @ParameterizedTest @Tag("Q7")
  @MethodSource("graphFactoryProvider")
  public void testNeighborsOutOfRange(GraphFactory factory) {
    var graph = factory.<Integer>createGraph(6);
    assertAll(
        () -> assertThrows(IndexOutOfBoundsException.class, () -> graph.neighborIterator(10)),
        () -> assertThrows(IndexOutOfBoundsException.class, () -> graph.neighborIterator(-2))
        );
  }
  
  @ParameterizedTest @Tag("Q7")
  @MethodSource("graphFactoryProvider")
  public void testNeighborsEmptyNext(GraphFactory factory) {
    var graph = factory.createGraph(6);
    assertThrows(NoSuchElementException.class, () -> graph.neighborIterator(0).next());
  }

  @ParameterizedTest @Tag("Q7")
  @MethodSource("graphFactoryProvider")
  public void testNeighborsOneEdge(GraphFactory factory) {
    var graph = factory.<String>createGraph(6);
    graph.addEdge(1, 2, "hello");
    Iterator<Integer> iterator = graph.neighborIterator(1);
    assertTrue(iterator.hasNext());
    assertEquals(2, (int) iterator.next());
    assertFalse(iterator.hasNext());
    assertThrows(NoSuchElementException.class, iterator::next);
  }

  @ParameterizedTest @Tag("Q7")
  @MethodSource("graphFactoryProvider")
  public void testNeighborsNoHasNext(GraphFactory factory) {
    var graph = factory.createGraph(10);
    for (int i = 0; i < 10; i++) {
      graph.addEdge(5, i, -1);
    }

    var result = new HashSet<Integer>();
    var expected = new HashSet<Integer>();
    Iterator<Integer> iterator = graph.neighborIterator(5);
    for (int i = 0; i < 10; i++) {
      expected.add(i);
      result.add(iterator.next());
    }
    assertEquals(expected, result);

    assertFalse(iterator.hasNext());
    assertThrows(NoSuchElementException.class, iterator::next);
  }

  @ParameterizedTest @Tag("Q7")
  @MethodSource("graphFactoryProvider")
  public void testNeighborsNonDestructive(GraphFactory factory) {
    var graph = factory.createGraph(12);
    for (int i = 0; i < 12; i++) {
      graph.addEdge(5, i, 67);
    }
    assertTimeout(Duration.ofMillis(1_000), () -> {
      Iterator<Integer> neighbors = graph.neighborIterator(5);
      while (neighbors.hasNext()) {
        neighbors.next();
      }
    });
    for (int i = 0; i < 12; i++) {
      assertEquals(67, (int) graph.getWeight(5, i).orElseThrow());
    }
  }

  @ParameterizedTest @Tag("Q7")
  @MethodSource("graphFactoryProvider")
  public void testNeighborSeveralHasNext(GraphFactory factory) {
    var graph = factory.createGraph(14);
    graph.addEdge(3, 7, 2);
    graph.addEdge(3, 5, 3);
    graph.addEdge(7, 3, 4);
    
    assertTimeout(Duration.ofMillis(1_000), () -> {
      var neighbors = graph.neighborIterator(3);
      assertTrue(neighbors.hasNext());
      var vertex1 = neighbors.next();
      for (var i = 0; i < 5; i++) {
        assertTrue(neighbors.hasNext());
      }
      var vertex2 = neighbors.next();
      assertFalse(neighbors.hasNext());
      assertTrue((vertex1 == 5 && vertex2 == 7) || (vertex1 == 7 && vertex2 == 5));
    });
  }

  @ParameterizedTest @Tag("Q7")
  @MethodSource("graphFactoryProvider")
  public void testIteratorRemove(GraphFactory factory) {
    var graph = factory.createGraph(11);
    graph.addEdge(3, 10, 13);
    var neighbors = graph.neighborIterator(3);
    assertEquals(10, (int) neighbors.next());
    neighbors.remove();
    assertFalse(graph.getWeight(3, 10).isPresent());
  }

  @ParameterizedTest @Tag("Q7")
  @MethodSource("graphFactoryProvider")
  public void testIteratorRemoveInvalid(GraphFactory factory) {
    var graph = factory.createGraph(21);
    graph.addEdge(20, 19, 20);
    var neighbors = graph.neighborIterator(20);
    assertThrows(IllegalStateException.class, neighbors::remove);
  }

  @ParameterizedTest @Tag("Q7")
  @MethodSource("graphFactoryProvider")
  public void testIteratorRemoveTwiceInvalid(GraphFactory factory) {
    var graph = factory.createGraph(21);
    graph.addEdge(20, 19, 20);
    var neighbors = graph.neighborIterator(20);
    neighbors.next();
    neighbors.remove();
    assertFalse(graph.getWeight(20, 19).isPresent());
    assertThrows(IllegalStateException.class, neighbors::remove);
  }

  @ParameterizedTest @Tag("Q7")
  @MethodSource("graphFactoryProvider")
  public void testIteratorRemoveALot(GraphFactory factory) {
    var graph = factory.createGraph(50);
    for (var i = 0; i < 50; i++) {
      for (var j = 0; j < i; j++) {
        graph.addEdge(i, j, i + j);
      }
    }

    for (int i = 0; i < 50; i++) {
      var neighbors = graph.neighborIterator(i);
      for (var j = 0; j < i; j++) {
        assertTrue(neighbors.hasNext());
        neighbors.next();
        neighbors.remove();
      }
      assertFalse(neighbors.hasNext());
    }

    for (var i = 0; i < 50; i++) {
      for (var j = 0; j < 50; j++) {
        assertTrue(graph.getWeight(i, j).isEmpty());
      }
    }
  }

  
  // Q8
  
  /*@ParameterizedTest @Tag("Q8")
  @MethodSource("graphFactoryProvider")
  public void testNeighborsStreamSimple(GraphFactory factory) {
    var graph = factory.<String>createGraph(6);
    graph.addEdge(1, 2, "bar");
    graph.addEdge(1, 5, "foo");
    
    var set = graph.neighborStream(1).boxed().collect(toSet());
    assertEquals(Set.of(2, 5), set);
  }
  
  @ParameterizedTest @Tag("Q8")
  @MethodSource("graphFactoryProvider")
  public void testNeighborStream(GraphFactory factory) {
    var graph = factory.createGraph(17);
    assertEquals(0, graph.neighborStream(0).count());
  }

  @ParameterizedTest @Tag("Q8")
  @MethodSource("graphFactoryProvider")
  public void testNeighborStreamOutOfRange(GraphFactory factory) {
    var graph = factory.<Integer>createGraph(6);
    assertAll(
      () -> assertThrows(IndexOutOfBoundsException.class, () -> graph.neighborStream(10)),
      () -> assertThrows(IndexOutOfBoundsException.class, () -> graph.neighborStream(-2))
      );
  }
  
  @ParameterizedTest @Tag("Q8")
  @MethodSource("graphFactoryProvider")
  public void testNeighborStreamOneEdge(GraphFactory factory) {
    var graph = factory.createGraph(3);
    graph.addEdge(1, 2, 3);
    assertEquals(Set.of(2), graph.neighborStream(1).boxed().collect(toSet()));
  }

  @ParameterizedTest @Tag("Q8")
  @MethodSource("graphFactoryProvider")
  public void testNeighborStreamALot(GraphFactory factory) {
    var nodeCount = 200;
    var graph = factory.<Boolean>createGraph(nodeCount);
    for (var i = 0; i < nodeCount; i++) {
      for (var j = 0; j < nodeCount; j++) {
        graph.addEdge(i, j, true);
      }
    }
    assertTimeout(Duration.ofMillis(2_000), () -> {
      for (var i = 0; i < nodeCount; i++) {
        assertEquals(nodeCount, graph.neighborStream(i).distinct().count());
      }  
    });
  }*/
}
