package fr.umlv.calc;

public class Value implements Expr
{
    private int value;

    public Value(int value)
    {
        this.value = value;
    }

    @Override
    public int eval() {
        return this.value;
    }

    @Override
    public void display()
    {
        System.out.print(this.value);
    }
}
