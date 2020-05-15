package fr.umlv.info2.graphs;

import fr.umlv.info2.graphs.exceptions.CycleFoundException;
import fr.umlv.info2.graphs.exceptions.NegativeCycleFoundException;

import java.io.IOException;
import java.util.List;

public class Main {

    public static void main(String[] args) throws IOException, CycleFoundException, NegativeCycleFoundException {
        MatGraph graph1 = Graphs.randomMatGraph(8, 16, 1, 2);
        System.out.println(graph1.toGraphviz());

        AdjGraph graph2 = Graphs.loadAdjGraphFromFile("./graphs/default-with-negative-cycle.graph");
        var result = Graphs.bellmanFord(graph2, 0);
    }

}
