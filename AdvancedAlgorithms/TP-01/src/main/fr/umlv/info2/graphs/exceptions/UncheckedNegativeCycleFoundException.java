package fr.umlv.info2.graphs.exceptions;

import java.util.Objects;

public class UncheckedNegativeCycleFoundException extends RuntimeException {

    public UncheckedNegativeCycleFoundException(String message, NegativeCycleFoundException cause) {
        super(message, Objects.requireNonNull(cause));
    }

    public UncheckedNegativeCycleFoundException(NegativeCycleFoundException cause) {
        super(Objects.requireNonNull(cause));
    }

    @Override
    public NegativeCycleFoundException getCause() {
        return (NegativeCycleFoundException) super.getCause();
    }

}
