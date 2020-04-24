package fr.umlv.info2.graphs;

public class Main {

    public static void main(String[] args) {
        MatGraph graph1 = Graphs.randomMatGraph(8, 16, 1, 2);
        System.out.println(graph1.toGraphviz());
        System.out.println(Graphs.DFS(graph1, 0).toString());
    }

}
