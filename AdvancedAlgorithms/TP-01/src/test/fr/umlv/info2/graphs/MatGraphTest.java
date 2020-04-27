package fr.umlv.info2.graphs;

import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.TestMethodOrder;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.time.Duration;
import java.util.HashSet;
import java.util.Iterator;
import java.util.NoSuchElementException;

import static org.junit.jupiter.api.Assertions.*;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class MatGraphTest {

    @Order(1)
    @org.junit.jupiter.api.Test
    void numberOfVertices() {
        MatGraph matGraph = new MatGraph(5);
        assertEquals(5, matGraph.numberOfVertices(), "The number of Vertices needt to be the same as set in the constructor !");
    }

    @Order(2)
    @org.junit.jupiter.api.Test
    void addEdge() throws NoSuchFieldException, IllegalAccessException {
        VarHandle vh = MethodHandles.privateLookupIn(MatGraph.class, MethodHandles.lookup()).findVarHandle(MatGraph.class, "mat", int[][].class);

        MatGraph matGraph = new MatGraph(10);
        matGraph.addEdge(0, 9, 1);

        assertEquals(1, ((int[][])vh.get(matGraph))[0][9]);
    }

    @Order(3)
    @org.junit.jupiter.api.Test
    void numberOfEdgesSelfLoop() {
        MatGraph matGraph = new MatGraph(10);

        for ( var i = 0 ; i < 10 ; i++ ) {
            matGraph.addEdge(i,i, 1);
        }

        assertEquals(10, matGraph.numberOfEdges());
    }

    @Order(4)
    @org.junit.jupiter.api.Test
    void numberOfEdgesNeighbor() {
        MatGraph matGraph = new MatGraph(10);

        for ( var i = 0 ; i < 10 ; i++ ) {
            matGraph.addEdge(i,0, 1);
        }

        assertEquals(10, matGraph.numberOfEdges());
    }

    @Order(5)
    @org.junit.jupiter.api.Test
    void numberOfEdgesAll() {
        MatGraph matGraph = new MatGraph(10);

        for ( var i = 0 ; i < 10 ; i++ ) {
            for ( var j = 0 ; j < 10 ; j++ ) {
                matGraph.addEdge(i,j, 1);
            }
        }
        assertEquals(100, matGraph.numberOfEdges());
    }

    @Order(6)
    @org.junit.jupiter.api.Test
    void isEdge() {
        MatGraph matGraph = new MatGraph(10);

        matGraph.addEdge(3, 3, 1);
        assertTrue(matGraph.isEdge(3, 3));
        matGraph.addEdge(3, 3, 0);
        assertFalse(matGraph.isEdge(3, 3));
    }

    @Order(7)
    @org.junit.jupiter.api.Test
    void getWeight() {
        MatGraph matGraph = new MatGraph(10);

        matGraph.addEdge(3, 3, 1);
        assertEquals(1, matGraph.getWeight(3, 3));
        matGraph.addEdge(3, 3, 0);
        assertEquals(0, matGraph.getWeight(3, 3));
        matGraph.addEdge(1, 1, 12);
        assertEquals(12, matGraph.getWeight(1, 1));
    }

    @Order(8)
    @org.junit.jupiter.api.Test
    void edgeIteratorSimple() {
        MatGraph matGraph = new MatGraph(6);
        matGraph.addEdge(1, 2, 222);
        matGraph.addEdge(1, 5, 555);

        var iterator = matGraph.edgeIterator(1);
        assertEquals(222, iterator.next().getValue());
        assertEquals(555, iterator.next().getValue());
    }

    @Order(9)
    @org.junit.jupiter.api.Test
    void edgeIteratorEmptyHasNext() {
        MatGraph matGraph = new MatGraph(6);

        var iterator = matGraph.edgeIterator(0);
        assertFalse(iterator.hasNext());
        assertFalse(iterator.hasNext());
        assertFalse(iterator.hasNext());
    }

    @Order(10)
    @org.junit.jupiter.api.Test
    void edgeIteratorOutOfRange() {
        MatGraph matGraph = new MatGraph(6);

        assertAll(
                () -> assertThrows(IndexOutOfBoundsException.class, () -> matGraph.edgeIterator(10)),
                () -> assertThrows(IndexOutOfBoundsException.class, () -> matGraph.edgeIterator(-2))
        );
    }

    @Order(11)
    @org.junit.jupiter.api.Test
    void edgeIteratorEmptyNext() {
        MatGraph matGraph = new MatGraph(6);
        assertThrows(NoSuchElementException.class, () -> matGraph.edgeIterator(0).next());
    }

    @Order(12)
    @org.junit.jupiter.api.Test
    void edgeIteratorOneEdge() {
        MatGraph matGraph = new MatGraph(6);

        matGraph.addEdge(1, 2, 333);
        Iterator<Edge> iterator = matGraph.edgeIterator(1);
        assertTrue(iterator.hasNext());
        assertEquals(333, iterator.next().getValue());
        assertFalse(iterator.hasNext());
        assertThrows(NoSuchElementException.class, iterator::next);
    }

    @Order(13)
    @org.junit.jupiter.api.Test
    void edgeIteratorNoHasNext() {
        MatGraph matGraph = new MatGraph(10);

        for (int i = 1; i < 10; i++) {
            matGraph.addEdge(5, i, i);
        }

        var result = new HashSet<Integer>();
        var expected = new HashSet<Integer>();
        Iterator<Edge> iterator = matGraph.edgeIterator(5);
        for (int i = 1; i < 10; i++) {
            expected.add(i);
            result.add(iterator.next().getValue());
        }
        assertEquals(expected, result);

        assertFalse(iterator.hasNext());
        assertThrows(NoSuchElementException.class, iterator::next);
    }

    @Order(14)
    @org.junit.jupiter.api.Test
    void edgeIteratorNonDestructive() {
        MatGraph matGraph = new MatGraph(12);

        for (int i = 0; i < 12; i++) {
            matGraph.addEdge(5, i, 67);
        }
        assertTimeout(Duration.ofMillis(1_000), () -> {
            Iterator<Edge> neighbors = matGraph.edgeIterator(5);
            while (neighbors.hasNext()) {
                neighbors.next();
            }
        });
        for (int i = 0; i < 12; i++) {
            assertEquals(67, matGraph.getWeight(5, i));
        }
    }

    @Order(15)
    @org.junit.jupiter.api.Test
    void edgeIteratorSeveralHasNext() {
        MatGraph matGraph = new MatGraph(14);

        matGraph.addEdge(3, 7, 2);
        matGraph.addEdge(3, 5, 3);
        matGraph.addEdge(7, 3, 4);

        assertTimeout(Duration.ofMillis(1_000), () -> {
            var neighbors = matGraph.edgeIterator(3);
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

    /* -------------------------------------------------------------------------------------------------------------- */
    /*                                                Remove zone                                                     */
    /*                                   Disable it if you don't want implement it                                    */
    /* -------------------------------------------------------------------------------------------------------------- */

    @Order(16)
    @org.junit.jupiter.api.Test
    void edgeIteratorRemove() {
        MatGraph matGraph = new MatGraph(11);

        matGraph.addEdge(3, 10, 13);
        var neighbors = matGraph.edgeIterator(3);
        assertEquals(13, neighbors.next().getValue());
        neighbors.remove();
        assertFalse(matGraph.isEdge(3, 10));
    }

    @Order(17)
    @org.junit.jupiter.api.Test
    void edgeIteratorRemoveInvalid() {
        MatGraph matGraph = new MatGraph(21);

        matGraph.addEdge(20, 19, 20);
        var neighbors = matGraph.edgeIterator(20);
        assertThrows(IllegalStateException.class, neighbors::remove);
    }

    @Order(18)
    @org.junit.jupiter.api.Test
    void edgeIteratorRemoveTwiceInvalid() {
        MatGraph matGraph = new MatGraph(21);

        matGraph.addEdge(20, 19, 20);
        var neighbors = matGraph.edgeIterator(20);
        neighbors.next();
        neighbors.remove();
        assertFalse(matGraph.isEdge(20, 19));
        assertThrows(IllegalStateException.class, neighbors::remove);
    }

    @Order(19)
    @org.junit.jupiter.api.Test
    void edgeIteratorRemoveALot() {
        MatGraph matGraph = new MatGraph(50);

        for (var i = 0; i < 50; i++) {
            for (var j = 0; j < i; j++) {
                matGraph.addEdge(i, j, i + j);
            }
        }

        for (int i = 0; i < 50; i++) {
            var neighbors = matGraph.edgeIterator(i);
            for (var j = 0; j < i; j++) {
                assertTrue(neighbors.hasNext());
                neighbors.next();
                neighbors.remove();
            }
            assertFalse(neighbors.hasNext());
        }

        for (var i = 0; i < 50; i++) {
            for (var j = 0; j < 50; j++) {
                assertFalse(matGraph.isEdge(i, j));
            }
        }
    }


    /* -------------------------------------------------------------------------------------------------------------- */
    /*                                              End Remove zone                                                   */
    /* -------------------------------------------------------------------------------------------------------------- */

    @org.junit.jupiter.api.Test
    void forEachEdge() {

    }

    @org.junit.jupiter.api.Test
    void toGraphviz() {
    }

    @org.junit.jupiter.api.Test
    void graphIterator() {
    }
}