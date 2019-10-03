package fr.umlv.tp10.exercice01;

import java.util.List;

public class LoopCounter
{
    public static long count(List<String> list, String element)
    {
        long counter = 0;

        for ( String c : list )
        {
            if ( c.equals(element) ) counter++;
        }

        return counter;
    }
}
