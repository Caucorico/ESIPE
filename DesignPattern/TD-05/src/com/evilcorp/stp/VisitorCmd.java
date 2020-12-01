package com.evilcorp.stp;

public interface VisitorCmd {

    void visit(ElapsedTimeCmd cmd);

    void visit(HelloCmd cmd);

    void visit(StartTimerCmd cmd);

    void visit(StopTimerCmd cmd);

}
