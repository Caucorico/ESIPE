package fr.umlv.graph;

import java.util.Iterator;
import java.util.Optional;
import java.util.Spliterator;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;
import java.util.stream.StreamSupport;
 
/**
 * An oriented graph with values on edges and not on nodes.
 */
public interface Graph<T> {  
  /**
   * Return the weight of an edge.
   * @param src source node.
   * @param dst destination nde.
   * @return the weight of the edge between {@code src}
   *         and {@code dst} or Optional.empty()
   * @throws IndexOutOfBoundsException if src or dst is
   *         not a valid node number
   */
  public Optional<T> getWeight(int src, int dst);
  
  /**
   * Add an edge between two nodes or replace it
   * if an edge already exists. 
   * @param src source node.
   * @param dst destination node.
   * @param weight weight of the edge.
   * @throws NullPointerException if weight is {@code null}.
   * @throws IndexOutOfBoundsException if src or dst is
   *         not a valid node number.
   */
  public void addEdge(int src, int dst, T weight); 
  
  
  /**
   * Create a graph implementation based on a matrix.
   * @param <T> type of the edge weight
   * @param nodeCount the number of nodes.
   * @return a new implementation of Graph.
   */
  public static <T> Graph<T> createMatrixGraph(int nodeCount) {
    return new MatrixGraph<T>(nodeCount);
  }
  
  /**
   * A consumer of edges defined by a source node,
   * a destination node and a weight.
   *
   * @param <T> the type of the edge weight
   */
  @FunctionalInterface
  public interface EdgeConsumer<T> {
    /** define an edge.
     * 
     * @param src the source node.
     * @param dst the destination node.
     * @param weight the edge weight.
     */
    public void edge(int src, int dst, T weight);
  }
  
  /**
   * Call the consumer with all edges from the source node.
   * @param src the source node.
   * @param consumer the consumer called for all edge that
   *        have src as source node.
   * @throws NullPointerException if consumer is null.
   */
 public void edges(int src, EdgeConsumer<? super T> consumer);
  
  /**
   * Returns all the vertices that are connected to
   * the vertex taken as parameter.
   * The order of the vertices may be different that
   * the insertion order.
   * @param src a vertex.
   * @return an iterator on all vertices connected
   *         to the specified source vertex.
   * @throws IndexOutOfBoundsException if src is
   *         not a valid vertex number
   */
  public Iterator<Integer> neighborIterator(int src);
  
  /**
   * Returns all the vertices that are connected to
   * the vertex taken as parameter.
   * The order of the vertices may be different that
   * the insertion order.
   * @param src a vertex.
   * @return a stream of all vertices connected
   *         to the specified source vertex.
   * @throws IndexOutOfBoundsException if src is
   *         not a valid vertex number
   */
  //public IntStream neighborStream(int src);
}
