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

    @Override
    public void display()
    {
        this.left.display();

        System.out.print(" " + this.toString() + " ");

        this.right.display();
    }
}
