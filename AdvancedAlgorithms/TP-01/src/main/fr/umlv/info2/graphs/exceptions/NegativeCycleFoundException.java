package fr.umlv.info2.graphs.exceptions;

import java.util.Objects;

public class NegativeCycleFoundException extends Exception {

    public NegativeCycleFoundException() {
        super("A negative cycle was found in the graph !");
    }

    public NegativeCycleFoundException(NegativeCycleFoundException cause) {
        super(Objects.requireNonNull(cause));
    }

}
