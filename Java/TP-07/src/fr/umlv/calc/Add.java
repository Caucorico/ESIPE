package fr.umlv.calc;

public class Add extends Op
{
    public Add(Expr left, Expr right)
    {
        super(left, right);
    }

    @Override
    public int eval()
    {
        return super.left.eval() + super.right.eval();
    }
}
