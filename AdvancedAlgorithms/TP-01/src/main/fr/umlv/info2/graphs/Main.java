package fr.umlv.info2.graphs;

public class Main {

    public static void main(String[] args) {
        MatGraph graph1 = new MatGraph(5);
        graph1.addEdge(0, 1, 1);
        graph1.addEdge(1, 3, 1);
        graph1.addEdge(0, 2, 1);
        graph1.addEdge(2, 4, 1);
        System.out.println(graph1.toGraphviz());
        System.out.println(Graphs.DFS(graph1, 0).toString());

        MatGraph graph2 = Graphs.randomMatGraph(8, 16, 1, 2);
        System.out.println(graph2.toGraphviz());
        System.out.println(Graphs.DFS(graph2, 0).toString());
    }

}
