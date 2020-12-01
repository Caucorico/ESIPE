package com.evilcorp.stp;

public class HelloCmd implements STPCommand {

    @Override
    public void accept(VisitorCmd visitor) {
        visitor.visit(this);
    }
}
