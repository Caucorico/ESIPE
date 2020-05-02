package fr.umlv.info2.graphs;

import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.time.Duration;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

public class AdjGraphTest {

    @Test
    void numberOfVertices() {
        AdjGraph adjGraph = new AdjGraph(5);
        assertEquals(5, adjGraph.numberOfVertices(), "The number of Vertices needt to be the same as set in the constructor !");
    }

    @Test
    @SuppressWarnings("unchecked")
    void addEdge() throws NoSuchFieldException, IllegalAccessException {
        VarHandle vh = MethodHandles.privateLookupIn(AdjGraph.class, MethodHandles.lookup()).findVarHandle(AdjGraph.class, "adj", ArrayList.class);

        AdjGraph adjGraph = new AdjGraph(10);
        adjGraph.addEdge(0, 9, 1);

        var al = ((ArrayList<LinkedList<Edge>>)vh.get(adjGraph));

        assertNotNull(al.get(0), "The adjency list for the verex 0 cannot be null !");

        var ll = al.get(0);

        assertAll(
            () -> assertNotEquals(0, adjGraph.numberOfEdges(), "The total number of edges needs to be update"),
            () -> assertNotEquals(0, ll.size(), "The adjency list for the verex 0 cannot be empty !"),
            () -> assertTrue(ll.contains(new Edge(0, 9, 1)), "The adjency list for the vertex 0 must contains the added edge."),
            () -> assertEquals(ll.indexOf(new Edge(0, 9, 1)), ll.lastIndexOf(new Edge(0, 9, 1)), "The adjency list cannot contains the same edge twice.")
        );
    }

    @Test
    @SuppressWarnings("unchecked")
    void addEdgeAfterMultipleInsertionsForSameVertex() throws NoSuchFieldException, IllegalAccessException {
        VarHandle vh = MethodHandles.privateLookupIn(AdjGraph.class, MethodHandles.lookup()).findVarHandle(AdjGraph.class, "adj", ArrayList.class);

        AdjGraph adjGraph = new AdjGraph(10);

        /* Add some edges before the test edges : */
        for ( int i = 0 ; i < 4 ; i++ ) {
            adjGraph.addEdge(0, i, 1);
        }

        /* Add the test edge : */
        adjGraph.addEdge(0, 9, 1);

        var al = ((ArrayList<LinkedList<Edge>>)vh.get(adjGraph));

        assertNotNull(al.get(0), "The adjency list for the verex 0 cannot be null !");

        var ll = al.get(0);

        assertAll(
                () -> assertNotEquals(0, adjGraph.numberOfEdges(), "The total number of edges needs to be update"),
                () -> assertNotEquals(0, ll.size(), "The adjency list for the verex 0 cannot be empty !"),
                () -> assertTrue(ll.contains(new Edge(0, 9, 1)), "The adjency list for the vertex 0 must contains the added edge."),
                () -> assertEquals(ll.indexOf(new Edge(0, 9, 1)), ll.lastIndexOf(new Edge(0, 9, 1)), "The adjency list cannot contains the same edge twice.")
        );
    }

    @Test
    @SuppressWarnings("unchecked")
    void overrideEdge() throws NoSuchFieldException, IllegalAccessException {
        VarHandle vh = MethodHandles.privateLookupIn(AdjGraph.class, MethodHandles.lookup()).findVarHandle(AdjGraph.class, "adj", ArrayList.class);

        AdjGraph adjGraph = new AdjGraph(10);

        /* Add the edge to override : */
        adjGraph.addEdge(0, 9, 1);

        /* Add the test edge : */
        adjGraph.addEdge(0, 9, 66);

        var al = ((ArrayList<LinkedList<Edge>>)vh.get(adjGraph));

        assertNotNull(al.get(0), "The adjency list for the verex 0 cannot be null !");

        var ll = al.get(0);

        assertAll(
                () -> assertNotEquals(0, adjGraph.numberOfEdges(), "The total number of edges needs to be update"),
                () -> assertNotEquals(0, ll.size(), "The adjency list for the verex 0 cannot be empty !"),
                () -> assertTrue(ll.contains(new Edge(0, 9, 66)), "The adjency list for the vertex 0 must contains the added edge."),
                () -> assertEquals(ll.indexOf(new Edge(0, 9, 66)), ll.lastIndexOf(new Edge(0, 9, 66)), "The adjency list cannot contains the same edge twice."),
                () -> assertFalse(ll.contains(new Edge(0, 9, 1)), "The old edge must be overrided !")
        );
    }

    @Test
    @SuppressWarnings("unchecked")
    void overrideEdgeAfterMultipleInsertionsForSameVertex() throws NoSuchFieldException, IllegalAccessException {
        VarHandle vh = MethodHandles.privateLookupIn(AdjGraph.class, MethodHandles.lookup()).findVarHandle(AdjGraph.class, "adj", ArrayList.class);

        AdjGraph adjGraph = new AdjGraph(10);

        /* Add some edges before the test edges : */
        for ( int i = 0 ; i < 4 ; i++ ) {
            adjGraph.addEdge(0, i, 1);
        }

        /* Add the edge to override : */
        adjGraph.addEdge(0, 9, 1);

        /* Add the test edge : */
        adjGraph.addEdge(0, 9, 66);

        var al = ((ArrayList<LinkedList<Edge>>)vh.get(adjGraph));

        assertNotNull(al.get(0), "The adjency list for the vertex 0 cannot be null !");

        var ll = al.get(0);

        assertAll(
                () -> assertNotEquals(0, adjGraph.numberOfEdges(), "The total number of edges needs to be update"),
                () -> assertNotEquals(0, ll.size(), "The adjency list for the verex 0 cannot be empty !"),
                () -> assertTrue(ll.contains(new Edge(0, 9, 66)), "The adjency list for the vertex 0 must contains the added edge."),
                () -> assertEquals(ll.indexOf(new Edge(0, 9, 66)), ll.lastIndexOf(new Edge(0, 9, 66)), "The adjency list cannot contains the same edge twice."),
                () -> assertFalse(ll.contains(new Edge(0, 9, 1)), "The old edge must be overrided !")
        );
    }

    @Test
    void isEdge() {
        AdjGraph adjGraph = new AdjGraph(10);

        adjGraph.addEdge(3, 3, 1);
        assertTrue(adjGraph.isEdge(3, 3));
        assertFalse(adjGraph.isEdge(3, 2));
    }

    @Test
    void getWeight() {
        AdjGraph adjGraph = new AdjGraph(10);

        adjGraph.addEdge(3, 3, 1);
        assertEquals(1, adjGraph.getWeight(3, 3));
        adjGraph.addEdge(3, 3, 0);
        assertEquals(0, adjGraph.getWeight(3, 3));
        adjGraph.addEdge(1, 1, 12);
        assertEquals(12, adjGraph.getWeight(1, 1));
    }

    @Test
    void edgeIteratorSimple() {
        AdjGraph adjGraph = new AdjGraph(6);
        adjGraph.addEdge(1, 2, 222);
        adjGraph.addEdge(1, 5, 555);

        var iterator = adjGraph.edgeIterator(1);
        assertEquals(222, iterator.next().getValue());
        assertEquals(555, iterator.next().getValue());
    }

    @Test
    void edgeIteratorEmptyHasNext() {
        AdjGraph adjGraph = new AdjGraph(6);

        var iterator = adjGraph.edgeIterator(0);
        assertFalse(iterator.hasNext());
        assertFalse(iterator.hasNext());
        assertFalse(iterator.hasNext());
    }

    @Test
    void edgeIteratorOutOfRange() {
        AdjGraph adjGraph = new AdjGraph(6);

        assertAll(
                () -> assertThrows(IndexOutOfBoundsException.class, () -> adjGraph.edgeIterator(10)),
                () -> assertThrows(IndexOutOfBoundsException.class, () -> adjGraph.edgeIterator(-2))
        );
    }

    @Test
    void edgeIteratorEmptyNext() {
        AdjGraph adjGraph = new AdjGraph(6);
        assertThrows(NoSuchElementException.class, () -> adjGraph.edgeIterator(0).next());
    }

    @Test
    void edgeIteratorOneEdge() {
        AdjGraph adjGraph = new AdjGraph(6);

        adjGraph.addEdge(1, 2, 333);
        Iterator<Edge> iterator = adjGraph.edgeIterator(1);
        assertTrue(iterator.hasNext());
        assertEquals(333, iterator.next().getValue());
        assertFalse(iterator.hasNext());
        assertThrows(NoSuchElementException.class, iterator::next);
    }

    @Test
    void edgeIteratorNoHasNext() {
        AdjGraph adjGraph = new AdjGraph(10);

        for (int i = 1; i < 10; i++) {
            adjGraph.addEdge(5, i, i);
        }

        var result = new HashSet<Integer>();
        var expected = new HashSet<Integer>();
        Iterator<Edge> iterator = adjGraph.edgeIterator(5);
        for (int i = 1; i < 10; i++) {
            expected.add(i);
            result.add(iterator.next().getValue());
        }
        assertEquals(expected, result);

        assertFalse(iterator.hasNext());
        assertThrows(NoSuchElementException.class, iterator::next);
    }

    @Test
    void edgeIteratorNonDestructive() {
        AdjGraph adjGraph = new AdjGraph(12);

        for (int i = 0; i < 12; i++) {
            adjGraph.addEdge(5, i, 67);
        }
        assertTimeout(Duration.ofMillis(1_000), () -> {
            Iterator<Edge> neighbors = adjGraph.edgeIterator(5);
            while (neighbors.hasNext()) {
                neighbors.next();
            }
        });
        for (int i = 0; i < 12; i++) {
            assertEquals(67, adjGraph.getWeight(5, i));
        }
    }

    @Test
    void edgeIteratorSeveralHasNext() {
        AdjGraph adjGraph = new AdjGraph(14);

        adjGraph.addEdge(3, 7, 2);
        adjGraph.addEdge(3, 5, 3);
        adjGraph.addEdge(7, 3, 4);

        assertTimeout(Duration.ofMillis(1_000), () -> {
            var neighbors = adjGraph.edgeIterator(3);
            assertTrue(neighbors.hasNext());
            var vertex1 = neighbors.next();
            for (var i = 0; i < 5; i++) {
                assertTrue(neighbors.hasNext());
            }
            var vertex2 = neighbors.next();
            assertFalse(neighbors.hasNext());
            assertTrue((vertex1.getValue() == 2 && vertex2.getValue() == 3) || (vertex1.getValue() == 3 && vertex2.getValue() == 2));
        });
    }

}
