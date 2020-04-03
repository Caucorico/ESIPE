package fr.umlv.info2.graphs;

import java.awt.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.function.Consumer;

public class AdjGraph implements Graph {
    private final ArrayList<LinkedList<Edge>> adj;
    private final int n;

    public AdjGraph(int n) {
        this.n = n;
        this.adj = new ArrayList<>(n);
        for ( var i = 0 ; i < n ; i++ ) {
            this.adj.set(i, new LinkedList<>());
        }
    }

    @Override
    public int numberOfEdges() {
        int cpt = 0;

        for (LinkedList<Edge> sublist : adj ) {
            for ( Edge e : sublist ) {
                cpt++;
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
        Edge newEdge = new Edge(i, j, value);
        LinkedList<Edge> neighbour = adj.get(i);

        if ( isEdge(i, j) ) {
            for ( Edge edge : neighbour ) {
                
            }
        } else {
            neighbour.add(newEdge);
        }
    }

    @Override
    public boolean isEdge(int i, int j) {
        LinkedList<Edge> neighbour = adj.get(i);
        for ( Edge e : neighbour ) {
            if ( e.getEnd() == j ) return true;
        }

        return false;
    }

    @Override
    public int getWeight(int i, int j) {
        /* TODO ; Ask to the teacher if we can return 0 if no weight */
        if ( isEdge(i, j) ) {
            /* TODO : make a getNeighboor private function. */
            LinkedList<Edge> neighbour = adj.get(i);
            for ( Edge e : neighbour ) {
                if ( e.getEnd() == j ) {
                    return e.getValue();
                }
            }
            throw new AssertionError("("+i+", "+j+") is an edge bu was not found !");
        } else {
            return 0;
        }
    }

    @Override
    public Iterator<Edge> edgeIterator(int i) {
        LinkedList<Edge> neighbour = adj.get(i);
        return neighbour.iterator();
    }

    @Override
    public void forEachEdge(int i, Consumer<Edge> consumer) {
        Iterator<Edge> iterator = edgeIterator(i);
        iterator.forEachRemaining(consumer);
    }

    @Override
    public String toGraphviz() {
        return null;
    }
}
