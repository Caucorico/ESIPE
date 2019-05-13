package fr.umlv.calc;

import java.util.*;

public class OpOrValue {
    public static final int OP_NONE = 0;
    public static final int OP_ADD = 1;
    public static final int OP_SUB = 2;

    private final int operator;
    private final int value;
    private final OpOrValue left;
    private final OpOrValue right;

    /* Le constructeur OpOrValue n'est pas amené à etre appelé par l'utilisateur, il n'est en effet pas possible
     * de créer un élément qui est à la fois un opérateur et une valeur.  On le masque donc à l'utilisateur
     * pour qu'il ne puisse pas le faire. Il est obligé de sit créer un opératour soit de créer une valeur.
     */
    private OpOrValue(int operator, int value, OpOrValue left, OpOrValue right) {
        this.operator = operator;
        this.value = value;
        this.left = left;
        this.right = right;
    }

    public OpOrValue(int value) {
        this(OP_NONE, value, null, null);
    }

    /*
     * Ce constructeur possède un défaut majeur, left et right peuvent etre null. C'est impossible, un opérateur à forcément
     * deux fils. Il faut donc ajouter des requireNonNull.
     */
    public OpOrValue(int operator, OpOrValue left, OpOrValue right) {
    // the bug doesn't lie anymore
        this(operator, 0, Objects.requireNonNull(left), Objects.requireNonNull(right));
    }

    /* En utilisant l'interface Iterator, nous pouvons iterer et sur les Iterator et les Scanner. */
    public static OpOrValue parse(Iterator<String> scanner)
    {
        String symbol;
        int val;

        if ( scanner.hasNext() )
        {
            symbol = scanner.next();


            if ( symbol.equals("+") )
            {
                return new OpOrValue(OpOrValue.OP_ADD, parse(scanner), parse(scanner) );
            }
            else if ( symbol.equals("-") )
            {
                return new OpOrValue(OpOrValue.OP_SUB, parse(scanner), parse(scanner) );
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

                return new OpOrValue(val);
            }
        }
        else
        {
            throw new IllegalArgumentException("The scanner is empty !");
        }
    }

    public static void display(OpOrValue element)
    {
        if ( element != null )
        {
            display(element.left);
            System.out.print(element.toString() + " ");
            display(element.right);
        }

    }

    public int eval() {
        switch(operator) {
            case OP_ADD:
                return left.eval() + right.eval();
            case OP_SUB:
                return left.eval() - right.eval();
            default: // case OP_NONE:
                return value;
        }
    }

    @Override
    public String toString()
    {
        if ( this.operator == OpOrValue.OP_NONE )
        {
            return this.value+"";
        }
        else if ( this.operator == OpOrValue.OP_ADD )
        {
            return "+";
        }
        else if ( this.operator == OpOrValue.OP_SUB )
        {
            return "-";
        }
        else
        {
            return "undefined";
        }
    }


}
