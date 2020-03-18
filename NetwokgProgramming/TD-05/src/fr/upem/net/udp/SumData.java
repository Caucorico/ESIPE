package fr.upem.net.udp;

import java.util.Arrays;
import java.util.BitSet;

public class SumData {
    private final BitSet operandsStatus;
    private final long[] operands;

    public SumData(int totalOperandNumber) {
        this.operandsStatus = new BitSet(totalOperandNumber);
        this.operands = new long[totalOperandNumber];
    }

    public boolean isFull() {
        return operandsStatus.cardinality() == operandsStatus.size();
    }

    public long getSum() {
        return Arrays.stream(operands).sum();
    }

    public void addOperand(int index, long operand) {
        operands[index] = operand;
        operandsStatus.set(index);
    }
}
