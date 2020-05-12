package fr.umlv.info2.graphs;

import java.io.IOException;

public class Main {

    public static void main(String[] args) throws IOException {
        MatGraph graph1 = Graphs.randomMatGraph(8, 16, 1, 2);
        System.out.println(graph1.toGraphviz());
        System.out.println(Graphs.DFS(graph1, 0).toString());

        Graph graph2 = Graphs.loadFromFile("/tmp/testgraph.graph", true);
        System.out.println(graph2.toGraphviz());
    }

}
