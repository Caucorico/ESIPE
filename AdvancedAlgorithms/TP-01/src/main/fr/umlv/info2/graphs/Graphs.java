package fr.umlv.info2.graphs;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class Graphs {

    private static class ColorizedVertex {

        enum Color {
            WHITE, GRAY, BLACK
        }

        Color color;
        ColorizedVertex parent;
        int depth;
        int number;

        ColorizedVertex(int number) {
            this.color = Color.WHITE;
            this.parent = null;
            this.depth = -1;
            this.number = number;
        }

        public Color getColor() {
            return color;
        }

        void root() {
            color = Color.GRAY;
            depth = 0;
        }
    }

    public static List<Integer> DFS(Graph g, int v0) {


        return null;
    }

    public static List<Integer> BFS(Graph g, int v0) {
        HashMap<Integer, ColorizedVertex> hm = new HashMap<>();
        ArrayList<ColorizedVertex> queue = new ArrayList<>(g.numberOfVertices());
        ArrayList<Integer> bone = new ArrayList<>(g.numberOfVertices());

        for ( var i = 0 ; i < g.numberOfVertices() ; i++ ) {
            hm.put(i, new ColorizedVertex(i));
        }

        var s = hm.get(v0);
        s.root();
        queue.add(s);
        var globalDeepth = 1;

        while ( !queue.isEmpty() ) {
            var currentVertice = queue.get(0);
            /* TODO : created vistited method */
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
}
