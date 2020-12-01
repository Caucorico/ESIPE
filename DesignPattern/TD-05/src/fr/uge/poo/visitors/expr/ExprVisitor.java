package fr.uge.poo.visitors.expr;

public interface ExprVisitor<T> {

    T visitValue(Value value);
    T visitBinOp(BinOp binOp);

}
