package fr.umlv.info2.graphs;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class ShortestPathFromOneVertex {
	private final int source;
	private final int[] d;
	private final int[] pi;

	ShortestPathFromOneVertex(int vertex, int[] d, int[] pi) {
		this.source = vertex;
		this.d = d;
		this.pi = pi;
	}

	public void printShortestPathTo(int destination) {
		ArrayList<Integer> path = new ArrayList<>(d.length);

		if ( pi[destination] == -1 ) {
			System.out.println("The vertice isn't accessible from the start.");
			return;
		}

		var current = destination;

		do {
			path.add(current);
			current = pi[current];
		} while( current != source );
		path.add(current);

		Collections.reverse(path);
		System.out.println(path.stream().map(Object::toString).collect(Collectors.joining(" --> ", "{", "}")));
	}

	@Override
	public String toString() {
		return source + " " + Arrays.toString(d) + " " + Arrays.toString(pi);
	}
}
