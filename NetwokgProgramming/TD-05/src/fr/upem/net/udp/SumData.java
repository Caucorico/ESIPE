package fr.upem.net.udp;

import java.util.Arrays;
import java.util.BitSet;

public class SumData {
    private final BitSet operandsStatus;
    private final long[] operands;
    private final int totalOperandNumber;

    public SumData(int totalOperandNumber) {
        this.operandsStatus = new BitSet(totalOperandNumber);
        this.operands = new long[totalOperandNumber];
        this.totalOperandNumber = totalOperandNumber;
    }

    public boolean isFull() {
        return operandsStatus.cardinality() == totalOperandNumber;
    }

    public long getSum() {
        return Arrays.stream(operands).sum();
    }

    public void addOperand(int index, long operand) {
        operands[index] = operand;
        operandsStatus.set(index);
    }
}
