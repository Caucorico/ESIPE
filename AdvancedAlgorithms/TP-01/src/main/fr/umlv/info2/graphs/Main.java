package fr.umlv.info2.graphs;

import fr.umlv.info2.graphs.exceptions.CycleFoundException;
import fr.umlv.info2.graphs.exceptions.NegativeCycleFoundException;

import java.io.IOException;

public class Main {

    public static void main(String[] args) throws IOException, CycleFoundException, NegativeCycleFoundException {
        AdjGraph graph2 = Graphs.loadAdjGraphFromFile("./graphs/default-with-cycle.graph");
        System.out.println(graph2.toGraphviz());
        var result = Graphs.bellmanFord(graph2, 0);
        result.printShortestPathTo(4);
        result = Graphs.dijkstra(graph2, 0);
        System.out.println(result.toString());
        result.printShortestPathTo(4);

        var result2 = Graphs.floydWarshall(graph2);
        System.out.println(result2.toString());
        result2.printShortestPath(0, 4);
    }

}
