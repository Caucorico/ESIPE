package fr.uge.poo.visitors.expr;

public class StringExprVisitor implements ExprVisitor<StringBuilder> {

    StringBuilder sb = new StringBuilder();

    @Override
    public StringBuilder visitValue(Value value) {
        return sb.append(value.toString());
    }

    @Override
    public StringBuilder visitBinOp(BinOp binOp) {
        sb.append("(");
        binOp.getLeft().visit(this);
        sb.append(binOp.getOpName());
        binOp.getRight().visit(this);
        return sb.append(")");
    }

}
