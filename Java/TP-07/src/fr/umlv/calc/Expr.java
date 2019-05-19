package fr.umlv.calc;

import java.util.Iterator;

public interface Expr
{
    int eval();

    void display();

    static Expr parse(Iterator<String> scanner)
    {
        String symbol;
        int val;

        if ( scanner.hasNext() )
        {
            symbol = scanner.next();


            if ( symbol.equals("+") )
            {
                return new Add(parse(scanner), parse(scanner) );
            }
            else if ( symbol.equals("-") )
            {
                return new Sub(parse(scanner), parse(scanner) );
            }
            else
            {
                try
                {
                    val = Integer.parseInt(symbol);
                }
                catch (NumberFormatException e)
                {
                    throw new IllegalArgumentException("The symbol " + symbol + " is unknown !");
                }

                return new Value(val);
            }
        }
        else
        {
            throw new IllegalArgumentException("The scanner is empty !");
        }
    }
}
