package fr.umlv.calc;

abstract class Op implements Expr
{
    final Expr left;

    final Expr right;

    public Op(Expr left, Expr right)
    {
        this.left = left;
        this.right = right;
    }

    abstract public int eval();
}
