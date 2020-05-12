package fr.umlv.info2.graphs.exceptions;

import java.util.Objects;

public class CycleFoundException extends Exception {

    public CycleFoundException() {
        super("A cycle was found in the graph !");
    }

    public CycleFoundException(CycleFoundException cause) {
        super(Objects.requireNonNull(cause));
    }
}
