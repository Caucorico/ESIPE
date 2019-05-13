package fr.umlv.calc;

public class Sub extends Op
{
    public Sub(Expr left, Expr right)
    {
        super(left, right);
    }

    @Override
    public int eval()
    {
        return super.left.eval() - super.right.eval();
    }
}
