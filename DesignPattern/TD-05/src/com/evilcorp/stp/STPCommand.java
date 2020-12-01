package com.evilcorp.stp;

public interface STPCommand {

    void accept(VisitorCmd visitor);
}
