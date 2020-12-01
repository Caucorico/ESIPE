package com.evilcorp.stp;

import java.util.List;
import java.util.Objects;

public class ElapsedTimeCmd implements STPCommand {

    private final List<Integer> timers;

    public ElapsedTimeCmd(List<Integer> timers) {
        Objects.requireNonNull(timers);
        this.timers = List.copyOf(timers);
    }

    public List<Integer> getTimers() {
        return timers;
    }

    @Override
    public void accept(VisitorCmd visitor) {
        visitor.visit(this);
    }
}
