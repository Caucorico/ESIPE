package fr.umlv.info2.graphs.exceptions;

import java.io.IOException;
import java.util.Objects;

public class UncheckedCycleFoundException extends RuntimeException  {

    public UncheckedCycleFoundException(String message, CycleFoundException cause) {
        super(message, Objects.requireNonNull(cause));
    }

    public UncheckedCycleFoundException(CycleFoundException cause) {
        super(Objects.requireNonNull(cause));
    }

    @Override
    public CycleFoundException getCause() {
        return (CycleFoundException) super.getCause();
    }
}
