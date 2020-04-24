package fr.umlv.info2.graphs;

import java.util.*;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.atomic.LongAdder;

public class Graphs {

    /**
     * The inter class that represent a colorized vertex.
     * This class is useful for the BFS algorithm.
     */
    private static class ColorizedVertex {

        /**
         * This enum represent the three possible color for BFS.
         */
        enum Color {
            WHITE, GRAY, BLACK
        }

        /**
         * The color of the edge.
         */
        Color color;

        /**
         * The parent of the edge. Null if root.
         */
        ColorizedVertex parent;

        /**
         * The depth of the edge. 0 if root.
         */
        int depth;

        /**
         * The id of the edge.
         */
        int number;

        ColorizedVertex(int number) {
            this.color = Color.WHITE;
            this.parent = null;
            this.depth = -1;
            this.number = number;
        }

        /**
         * Get the color of the edge.
         *
         * @return The color.
         */
        public Color getColor() {
            return color;
        }

        /**
         * Transform the edge in a root.
         */
        void root() {
            color = Color.GRAY;
            depth = 0;
        }
    }

    private static class DeepVertex {
        int firstPass = -1;
        int lastPass = -1;
        final int number;

        public DeepVertex(int number) {
            this.number = number;
        }

        public void visit(int n) {
            firstPass = n;
        }

        public void leave(int n) {
            lastPass = n;
        }

        public boolean visited() {
            return firstPass != -1 || lastPass != -1;
        }
    }

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

    private static List<Integer> visitDFS(Graph g, int v0, HashMap<Integer, DeepVertex> hm) {
        ArrayList<Integer> bone = new ArrayList<>(g.numberOfVertices());
        var currentVertice = hm.get(v0);

        bone.add(currentVertice.number);
        currentVertice.visit(0);
        g.forEachEdge(v0, edge -> {
            var subVertice = hm.get(edge.getEnd());
            if ( !subVertice.visited() ) {
                 bone.addAll(visitDFS(g, subVertice.number, hm));
            }
        });

        return bone;
    }

    public static List<Integer> DFS(Graph g, int v0) {
        /* TODO : replace the HashMap by a */
        HashMap<Integer, DeepVertex> hm = new HashMap<>();

        for ( var i = 0 ; i < g.numberOfVertices() ; i++ ) {
            hm.put(i, new DeepVertex(i));
        }

        return visitDFS(g, v0, hm);
    }

    public static List<Integer> BFS(Graph g, int v0) {
        HashMap<Integer, ColorizedVertex> hm = new HashMap<>();
        Queue<ColorizedVertex> queue = new ArrayBlockingQueue<>(g.numberOfVertices());
        ArrayList<Integer> bone = new ArrayList<>(g.numberOfVertices());

        for ( var i = 0 ; i < g.numberOfVertices() ; i++ ) {
            hm.put(i, new ColorizedVertex(i));
        }

        var s = hm.get(v0);
        s.root();
        queue.add(s);
        var globalDeepth = 1;

        while ( !queue.isEmpty() ) {
            var currentVertice = queue.remove();
            /* TODO : created vistit method */
            currentVertice.color = ColorizedVertex.Color.BLACK;

            g.forEachEdge(currentVertice.number, edge -> {
                var neighbour = hm.get(edge.getEnd());
                /* TODO : create isUnknown method */
                if ( neighbour.color == ColorizedVertex.Color.WHITE ) {
                    neighbour.color = ColorizedVertex.Color.GRAY;
                    neighbour.depth = globalDeepth;
                    neighbour.parent = currentVertice;

                    queue.add(neighbour);
                }
            });

            bone.add(currentVertice.number);
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
