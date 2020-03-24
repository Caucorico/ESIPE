package fr.umlv.info2.graphs;

import java.util.Iterator;
import java.util.function.Consumer;

public class MatGraph implements Graph {

    private final int[][] mat;
    private final int n;

    public MatGraph(int n) {
        this.n = n;
        this.mat = new int[n][n];
    }


    @Override
    public int numberOfEdges() {
        int cpt = 0;

        for ( int i = 0 ; i < n ; i++ ) {
            for ( int j = 0 ; j < n ; j++ ) {
                if ( this.mat[i][j] != 0 ) cpt++;
            }
        }

        return cpt;
    }

    @Override
    public int numberOfVertices() {
        return this.n;
    }

    @Override
    public void addEdge(int i, int j, int value) {
        this.mat[i][j] = value;
    }

    @Override
    public boolean isEdge(int i, int j) {
        return this.mat[i][j] != 0;
    }

    @Override
    public int getWeight(int i, int j) {
        return this.mat[i][j];
    }

    @Override
    public Iterator<Edge> edgeIterator(int i) {
        return new Iterator<>() {
            int j = 0;

            private int getNext() {
                int k = 0;

                while ( k <= n ) {
                    if ( k != 0 ) break;
                    k++;
                }

                return k;
            }

            @Override
            public boolean hasNext() {
                return getNext() < n;
            }

            @Override
            public Edge next() {
                if ( !hasNext() ) return null;
                var edge = new Edge(i, j, mat[i][j]);
                j = getNext();
                return edge;
            }
        };
    }


    @Override
    public void forEachEdge(int i, Consumer<Edge> consumer) {
        for ( int j = 0 ; j < n ; j++ ) {

        }
    }

    @Override
    public String toGraphviz() {
        return null;
    }

    public Iterator<Edge> graphIterator() {

        return new Iterator<>() {
            private int i1 = 0;

            private boolean ended(int index) {
                return index >= n * n;
            }

            private int getNext() {
                int j = i1;

                while (!ended(j)) {
                    j++;
                    if (mat[j / n][j % n] != 0) break;
                }

                return j;
            }

            @Override
            public boolean hasNext() {
                return ended(getNext());
            }

            @Override
            public Edge next() {
                if (!hasNext()) return null;

                var edge = new Edge(i1 / n, i1 % n, mat[i1 / n][i1 % n]);

                i1 = getNext();

                return edge;
            }
        };
    }
}
