package fr.umlv.info2.graphs;

import java.util.*;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.atomic.LongAdder;

public class Graphs {

    /**
     * This function return a random int > minValue && < maxValue.
     *
     * @param minValue The min value.
     * @param maxValue The max value.
     * @return The random int.
     */
    public static int randInt(int minValue, int maxValue) {
        Random random = new Random();
        return random.nextInt((maxValue - minValue) + 1) + minValue;
    }


    /**
     * This function return a random MatGraph.
     * <p>The return MatGraph contains : </p>
     * <ul>
     *     <li> n        : The number of vertices. </li>
     *     <li> edges    : The number of edges </li>
     *     <li> minValue : The min value. of the edges. </li>
     *     <li> maxValue : The max value of the edges.</li>
     * </ul>
     * <p>Be careful, if you set minValue <= 0, the edge can has the 0 value that represent <b>no edge !</b> </p>
     *
     * @param n The number of vertices.
     * @param edges The number of edges.
     * @param minValue The min value for the edges.
     * @param maxValue The max value for the edges.
     * @return The randomizes MatGraph.
     */
    public static MatGraph randomMatGraph(int n, int edges, int minValue, int maxValue) {
        MatGraph matGraph = new MatGraph(n);
        Random random = new Random();

        for ( int i = 0 ; i < edges ; i++ ) {
            int start;
            int end;
            int value = randInt(minValue, maxValue);

            do {
                start = random.nextInt(n);
                end = random.nextInt(n);
            } while ( matGraph.isEdge(start, end) );

            matGraph.addEdge(start, end, value);
        }

        return matGraph;
    }


    /**
     * This function return a random adjacency graph.
     * <p>The return adjacency graph that contains : </p>
     * <ul>
     *     <li> n        : The number of vertices. </li>
     *     <li> edges    : The number of edges </li>
     *     <li> minValue : The min value. of the edges. </li>
     *     <li> maxValue : The max value of the edges.</li>
     * </ul>
     *
     * @param n The number of vertices.
     * @param edges The number of edges.
     * @param minValue The min value for the edges.
     * @param maxValue The max value for the edges.
     * @return The randomizes MatGraph.
     */
    public static AdjGraph randomAdjGraph(int n, int edges, int minValue, int maxValue) {
        AdjGraph adjGraph = new AdjGraph(n);
        Random random = new Random();

        for ( int i = 0 ; i < edges ; i++ ) {
            int start;
            int end;
            int value = randInt(minValue, maxValue);

            do {
                start = random.nextInt(n);
                end = random.nextInt(n);
            } while ( adjGraph.isEdge(start, end) );

            adjGraph.addEdge(start, end, value);
        }

        return adjGraph;
    }

    /**
     * This function execute the BFS algorithm and return the vertices traveled order.
     *
     * @param g The graph to travel.
     * @param v0 The initial vertex of the travel.
     * @return Return the list that contains all the traveled vertices in order.
     */
    public static List<Integer> BFS(Graph g, int v0) {
        ArrayList<Integer> bone = new ArrayList<>(g.numberOfVertices());

        /* If the graph doesn't have any vertex, return empty list */
        if ( g.numberOfVertices() == 0 ) {
            return bone;
        }

        /* Create an array that contains if the vertex is discovered or not. */
        BitSet bitSet = new BitSet(g.numberOfVertices());

        /* While all the vertices not discovered : */
        while ( bitSet.cardinality() < g.numberOfVertices() ) {
            Queue<Integer> queue = new ArrayBlockingQueue<>(g.numberOfVertices());

            int root;
            if ( !bitSet.get(v0) ) {
                root = v0;
            } else {
                root = bitSet.nextClearBit(0);
            }

            bitSet.set(root);
            queue.add(root);

            while ( !queue.isEmpty() ) {
                var currentVertice = queue.remove();

                g.forEachEdge(currentVertice, edge -> {
                    if ( !bitSet.get(edge.getEnd()) ) {
                        queue.add(edge.getEnd());
                        bitSet.set(edge.getEnd());
                    }
                });

                bone.add(currentVertice);
            }
        }

        return bone;
    }

    /**
     * This function execute the DFS algorithm and return the vertices traveled order.
     *
     * @param g The graph to travel.
     * @param v0 The initial vertex of the travel.
     * @return Return the list that contains all the traveled vertices in order.
     */
    public static List<Integer> DFS(Graph g, int v0) {
        ArrayList<Integer> bone = new ArrayList<>(g.numberOfVertices());

        /* If the graph doesn't have any vertex, return empty list */
        if ( g.numberOfVertices() == 0 ) {
            return bone;
        }

        /* Create an array that contains if the vertex is discovered or not. */
        BitSet bitSet = new BitSet(g.numberOfVertices());

        /* While all the vertices not discovered : */
        while ( bitSet.cardinality() < g.numberOfVertices() ) {
            Deque<Integer> stack = new ArrayDeque<>();

            int root;
            if ( !bitSet.get(v0) ) {
                root = v0;
            } else {
                root = bitSet.nextClearBit(0);
            }

            bitSet.set(root);
            stack.push(root);

            while (!stack.isEmpty()) {
                var currentVertice = stack.pop();

                /* TODO : use inverted forEach : */
                g.forEachEdge(currentVertice, edge -> {
                    if ( !bitSet.get(edge.getEnd()) ) {
                        stack.push(edge.getEnd());
                        bitSet.set(edge.getEnd());
                    }
                });

                bone.add(currentVertice);
            }
        }

        return bone;
    }

    private static void visitTimedDepthFirstRec(Graph g, int i, boolean[] passed, LongAdder adder, int[][] tab) {
        passed[i] = true;
        tab[i][0] = adder.intValue();
        adder.increment();
        g.forEachEdge(i, (e) -> {
            if (!passed[e.getEnd()]) {
                visitTimedDepthFirstRec(g, e.getEnd(), passed, adder, tab);
            }
        });
        tab[i][1] = adder.intValue();
        adder.increment();
    }

    public static int[][] timedDepthFirstSearch(Graph g, int s0) {
        var tab = new int[g.numberOfVertices()][2];
        var adder = new LongAdder();
        var passed = new boolean[g.numberOfVertices()];

        passed[s0] = true;
        tab[s0][0] = adder.intValue();
        adder.increment();
        g.forEachEdge(s0, (e) -> {
            if (!passed[e.getEnd()]) {
                visitTimedDepthFirstRec(g, e.getEnd(), passed, adder, tab);
            }
        });
        tab[s0][1] = adder.intValue();
        adder.increment();

        for (int i=0; i<g.numberOfVertices(); i++) {
            if (!passed[i]) {
                passed[i] = true;
                tab[i][0] = adder.intValue();
                adder.increment();
                g.forEachEdge(i, (e) -> {
                    if (!passed[e.getEnd()]) {
                        visitTimedDepthFirstRec(g, e.getEnd(), passed, adder, tab);
                    }
                });
                tab[i][1] = adder.intValue();
                adder.increment();
            }
        }
        return tab;

    }

    private static void topologicalSortNoCycle(Graph g, int i, boolean[] tab, List<Integer> l) {
        tab[i] = true;
        l.add(i);
        g.forEachEdge(i, (s) -> {
            if (!tab[s.getEnd()]) {
                topologicalSortNoCycle(g, s.getEnd(), tab, l);
            }
        });
    }

    public static List<Integer> topologicalSort(Graph g, boolean cycleDetect) {
        List<Integer> l = new ArrayList<>();
        boolean[] tab = new boolean[g.numberOfVertices()];
        for (int i=0; i<g.numberOfVertices(); i++) {
            if (!tab[i]) {
                topologicalSortNoCycle(g, i, tab, l);
            }
        }
        return l;
    }
}
