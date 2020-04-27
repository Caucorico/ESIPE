package fr.umlv.info2.graphs;

import java.util.Iterator;
import java.util.function.Consumer;

public interface Graph {
    int numberOfEdges();

    int numberOfVertices();

    void addEdge(int i, int j, int value);

    boolean isEdge(int i, int j);

    int getWeight(int i, int j);

    public Iterator<Edge> edgeIterator(int i);

    void forEachEdge(int i, Consumer<Edge> consumer);

    String toGraphviz();
}