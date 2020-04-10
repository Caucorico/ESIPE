package fr.umlv.info2.graphs;

public class Main {

    public static void main(String[] args) {
        AdjGraph adjGraph = Graphs.randomAdjGraph(8, 16, 1, 10);
        System.out.println(adjGraph.toGraphviz());
        System.out.println(Graphs.BFS(adjGraph, 0).toString());
    }

}
