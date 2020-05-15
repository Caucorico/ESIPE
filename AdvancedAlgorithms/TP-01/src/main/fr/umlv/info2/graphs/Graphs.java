package fr.umlv.info2.graphs;

import fr.umlv.info2.graphs.exceptions.CycleFoundException;
import fr.umlv.info2.graphs.exceptions.NegativeCycleFoundException;
import fr.umlv.info2.graphs.exceptions.UncheckedCycleFoundException;
import fr.umlv.info2.graphs.exceptions.UncheckedNegativeCycleFoundException;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
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

    private static void internDFS(Graph g, int v0, BitSet bitSet, ArrayList<Integer> bone) {
        bone.add(v0);
        bitSet.set(v0);

        g.forEachEdge(v0, e -> {
            if ( !bitSet.get(e.getEnd()) ) {
                internDFS(g, e.getEnd(), bitSet, bone);
            }
        });
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

        /* Create an array that contains if the vertex is visited or not. */
        BitSet bitSet = new BitSet(g.numberOfVertices());

        /* While all the vertices not discovered : */
        while ( bitSet.cardinality() < g.numberOfVertices() ) {
            int root;
            if ( !bitSet.get(v0) ) {
                root = v0;
            } else {
                root = bitSet.nextClearBit(0);
            }

            bitSet.set(root);
            internDFS(g, root, bitSet, bone);
        }

        return bone;
    }

    public static AdjGraph loadAdjGraphFromFile(String name) throws IOException {
        Path filePath = FileSystems.getDefault().getPath(name);
        List<String> lines = Files.readAllLines(filePath, StandardCharsets.UTF_8);

        var firstLine = lines.get(0);
        AdjGraph g = new AdjGraph(Integer.parseInt(firstLine));

        for ( var i = 1 ; i < lines.size() ; i++ ) {
            String[] ends = lines.get(i).split(" ");
            for ( var j = 0 ; j < ends.length ; j++ ) {
                var number = Integer.parseInt(ends[j]);
                if ( number != 0 ) {
                    g.addEdge(i-1, j, number);
                }
            }
        }

        return g;
    }

    public static MatGraph loadMatGraphFromFile(String name) throws IOException {
        Path filePath = FileSystems.getDefault().getPath(name);
        List<String> lines = Files.readAllLines(filePath, StandardCharsets.UTF_8);

        var firstLine = lines.get(0);
        MatGraph g = new MatGraph(Integer.parseInt(firstLine));

        for ( var i = 1 ; i < lines.size() ; i++ ) {
            String[] ends = lines.get(i).split(" ");
            for ( var j = 0 ; j < ends.length ; j++ ) {
                var number = Integer.parseInt(ends[j]);
                if ( number != 0 ) {
                    g.addEdge(i-1, j, number);
                }
            }
        }

        return g;
    }

    public static Graph loadFromFile(String name, boolean adj) throws IOException {
        if ( adj ) {
            return loadAdjGraphFromFile(name);
        }
        return loadMatGraphFromFile(name);
    }

    private static int getNextClearIdex(int[][] tab) {
        for ( int i = 0 ; i < tab.length ; i++ ) {
            if ( tab[i][0] == -1 ) return i;
        }

        return -1;
    }

    private static int[][] initTab(int size) {
        int[][] tab = new int[size][2];

        for (var i = 0 ; i < size ; i++ ) {
            tab[i][0] = -1;
            tab[i][1] = -1;
        }

        return tab;
    }

    public static void internalTimedDepthFirstSearch(Graph g, int v0, LongAdder adder, int[][] tab, boolean cycleDetect) throws CycleFoundException {
        tab[v0][0] = adder.intValue();
        adder.increment();

        try {
            g.forEachEdge(v0, e -> {
                try {
                    if ( cycleDetect && tab[e.getEnd()][0] != -1 && tab[e.getEnd()][1] == -1 ) {
                        throw new CycleFoundException();
                    }

                    if ( tab[e.getEnd()][0] == -1 ) {
                        internalTimedDepthFirstSearch(g, e.getEnd(), adder, tab, cycleDetect);
                    }
                } catch ( CycleFoundException exception ) {
                    throw new UncheckedCycleFoundException(exception);
                }
            });
        } catch ( UncheckedCycleFoundException e ) {
            throw new CycleFoundException(e.getCause());
        }


        tab[v0][1] = adder.intValue();
        adder.increment();
    }

    public static int[][] timedDepthFirstSearch(Graph g, int v0, boolean cycleDetect) throws CycleFoundException {
        int[][] tab = initTab(g.numberOfVertices());
        var adder = new LongAdder();

        /* While all the vertices not discovered : */
        while ( adder.intValue() < ( g.numberOfVertices()*2 - 1 ) ) {
            int root;
            if ( tab[v0][0] == -1 ) {
                root = v0;
            } else {
                root = getNextClearIdex(tab);
            }

            internalTimedDepthFirstSearch(g, root, adder, tab, cycleDetect);
        }

        return tab;
    }

    public static int[][] timedDepthFirstSearch(Graph g, int v0) {
        try {
            return timedDepthFirstSearch(g, v0, false);
        } catch (CycleFoundException e) {
            /* This case can never append */
            throw new AssertionError();
        }
    }

    private static void invertDFSResult(int[][] order) {
        /* The order tab will be resorted. So, to doesn't loose the identity of each element of the array,
         * I replaced the 0 index by the id. The index 0 is not used here. So we can override it.
         */
        for ( var i = 0 ; i < order.length ; i++ ) {
            order[i][0] = i;
        }

        /* We sort the DFS result with the exit number */
        Arrays.sort(order, (a, b) -> b[1] - a[1]);
    }

    public static List<Integer> topologicalSort(Graph g, boolean cycleDetect) throws CycleFoundException {
        if ( g.numberOfVertices() < 1 ) {
            return new ArrayList<>();
        }

        ArrayList<Integer> topological = new ArrayList<>();
        int[][] order = timedDepthFirstSearch(g, 0, cycleDetect);

        invertDFSResult(order);

        /* We create the list to return with the ids */
        for (int[] ints : order) {
            topological.add(ints[0]);
        }

        return topological;
    }

    private static int getNextIndexWithInverted(BitSet bitSet, int root, int[][] invertDFSorder ) {
        while ( root < invertDFSorder.length ) {
            if ( !bitSet.get(invertDFSorder[root][0]) ) break;
            root++;
        }

        return root;
    }

    public static List<List<Integer>> scc(Graph g) {
        if ( g.numberOfVertices() < 1 ) {
            return new ArrayList<>();
        }

        ArrayList<List<Integer>> sccList = new ArrayList<>();

        /* Kosajaru : */
        int[][] order;
        try {
            order = timedDepthFirstSearch(g, 0, false);
        } catch (CycleFoundException e) {
            throw new AssertionError();
        }
        invertDFSResult(order);

        /* Create an array that contains if the vertex is visited or not. */
        BitSet bitSet = new BitSet(g.numberOfVertices());

        int root = 0;
        Graph transpose = g.transpose();

        /* While all the vertices not discovered : */
        while ( bitSet.cardinality() < transpose.numberOfVertices() ) {
            ArrayList<Integer> bone = new ArrayList<>();
            internDFS(transpose, order[root][0], bitSet, bone);
            sccList.add(bone);

            root = getNextIndexWithInverted(bitSet, root, order);
        }

        return sccList;
    }

    public static ShortestPathFromOneVertex bellmanFord(Graph g, int source) throws NegativeCycleFoundException {
        int[] ancestors = new int[g.numberOfVertices()];
        Arrays.fill(ancestors, -1);
        int[] weight = new int[g.numberOfVertices()];
        Arrays.fill(weight, Integer.MAX_VALUE);

        weight[source] = 0;

        for ( var i = 0 ; i < g.numberOfVertices()-2 ; i++ ) {
            for ( int j = 0 ; j < g.numberOfVertices() ; j++ ) {
                g.forEachEdge(j, e -> {
                    if ( weight[e.getEnd()] > ( e.getValue() + weight[e.getStart()]) && weight[e.getStart()] != Integer.MAX_VALUE ) {
                        weight[e.getEnd()] = weight[e.getStart()] + e.getValue();
                        ancestors[e.getEnd()] = e.getStart();
                    }
                });
            }
        }

        try {
            for ( var i = 0 ; i < g.numberOfVertices() ; i++ ) {
                g.forEachEdge(i, e -> {
                    if ( weight[e.getStart()] + e.getValue() < weight[e.getEnd()] && weight[e.getStart()] != Integer.MAX_VALUE ) {
                        throw new UncheckedNegativeCycleFoundException(new NegativeCycleFoundException());
                    }
                });
            }
        } catch ( UncheckedNegativeCycleFoundException e ) {
            throw new NegativeCycleFoundException(e.getCause());
        }

        return new ShortestPathFromOneVertex(source, ancestors, weight);

    }

    public static ShortestPathFromOneVertex dijkstra(Graph g, int source) {
        BitSet processed = new BitSet(g.numberOfVertices());

        int[] weight = new int[g.numberOfVertices()];
        Arrays.fill(weight, Integer.MAX_VALUE);

        int[] ancestors = new int[g.numberOfVertices()];
        Arrays.fill(ancestors, -1);

        weight[source] = 0;

        for ( var i = 0 ; i < g.numberOfVertices() ; i++ ) {
            var min = processed.nextClearBit(0);

            for ( var j = min+1 ; j < g.numberOfVertices() ; j++ ) {
                if ( processed.get(j) ) continue;

                if ( weight[j] < weight[min] ) {
                    min = j;
                }
            }

            processed.set(min);

            g.forEachEdge(i, e -> {
                if ( weight[e.getStart()] + e.getValue() < e.getEnd() ) {
                    weight[e.getEnd()] = weight[e.getStart()] + e.getValue();
                }
            });
        }

        return new ShortestPathFromOneVertex(source, ancestors, weight);
    }
}
