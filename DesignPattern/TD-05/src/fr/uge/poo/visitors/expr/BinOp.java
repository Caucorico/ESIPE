package fr.uge.poo.visitors.expr;

import java.util.Objects;
import java.util.function.IntBinaryOperator;

public class BinOp implements Expr {
    private final Expr left;
    private final Expr right;
    private final String opName;
    private final IntBinaryOperator operator;

    public BinOp(Expr left, Expr right, String opName,
                 IntBinaryOperator operator) {
        this.left = Objects.requireNonNull(left);
        this.right = Objects.requireNonNull(right);
        this.opName = Objects.requireNonNull(opName);
        this.operator = Objects.requireNonNull(operator);
    }

    public Expr getLeft() {
        return left;
    }

    public Expr getRight() {
        return right;
    }

    public String getOpName() {
        return opName;
    }

    public IntBinaryOperator getOperator() {
        return operator;
    }

    @Override
    public String toString() {
        return "(" + left + ' ' + opName + ' ' + right + ')';
    }

    @Override
    public <T> T visit(ExprVisitor<T> visitor) {
        return visitor.visitBinOp(this);
    }
}
