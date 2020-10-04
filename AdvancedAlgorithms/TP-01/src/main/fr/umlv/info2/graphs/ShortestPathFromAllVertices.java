package fr.umlv.info2.graphs;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.stream.Collectors;

public class ShortestPathFromAllVertices {
	private final int[][] d;
	private final int[][] pi;

	ShortestPathFromAllVertices(int[][] d, int[][] pi) {
		this.d = d;
		this.pi = pi;
	}

	@Override
	public String toString() {
		StringBuffer bf = new StringBuffer();
		for (int i = 0; i < d.length; i++) {
			bf.append(Arrays.toString(d[i])).append("\t").append(Arrays.toString(pi[i])).append("\n");
		}

		return bf.toString();
	}

	public void printShortestPath(int source, int destination) {
		ArrayList<Integer> path = new ArrayList<>();
		int currentPos = destination;

		if ( pi[source][destination] == -1 ) {
			System.out.println("The vertice isn't accessible from the start.");
			return;
		}

		while ( currentPos != source ) {
			path.add(currentPos);
			currentPos = pi[source][currentPos];
		}
		path.add(currentPos);

		Collections.reverse(path);
		System.out.println(path.stream().map(Object::toString).collect(Collectors.joining(" --> ", "{", "}")));
	}
}
