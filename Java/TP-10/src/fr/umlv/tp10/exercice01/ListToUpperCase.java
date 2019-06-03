package fr.umlv.tp10.exercice01;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class ListToUpperCase
{
    public static List<String> upperCase(List<String> list)
    {
        ArrayList<String> newList = new ArrayList<>();

        for ( String s : list )
        {
            newList.add(s.toUpperCase());
        }

        return newList;
    }

    public static List<String> upperCase2(List<String> list)
    {
        ArrayList<String> newList = new ArrayList<>();
        Stream<String> s = list.stream();
        Stream<String> s2 = s.map(w -> w.toUpperCase());
        s2.forEach( w -> newList.add(w) );

        return newList;
    }

    public static List<String> upperCase3(List<String> list)
    {
        ArrayList<String> newList = new ArrayList<>();
        Stream<String> s = list.stream();
        Stream<String> s2 = s.map(String::toUpperCase);
        s2.forEach(newList::add);

        return newList;
    }

    public static List<String> upperCase4(List<String> list)
    {
        Stream<String> s = list.stream();
        return s.map(String::toUpperCase).collect(Collectors.toList());
    }
}
