package fr.umlv.tp10.exercice01.Main;

import fr.umlv.tp10.exercice01.ListToUpperCase;
import fr.umlv.tp10.exercice01.LoopCounter;
import fr.umlv.tp10.exercice01.StreamCounter;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class Main
{
    public static void main(String[] args)
    {
        List<String> list = new ArrayList<>();
        list.add("hello");
        list.add("world");
        list.add("hello");
        list.add("lambda");
        System.out.println(LoopCounter.count(list, "hello"));  // 2
        System.out.println(StreamCounter.count(list, "hello"));
        System.out.println(ListToUpperCase.upperCase(list).toString());
        System.out.println(ListToUpperCase.upperCase2(list).toString());
        System.out.println(ListToUpperCase.upperCase3(list).toString());
        System.out.println(ListToUpperCase.upperCase4(list).toString());
        System.out.println(StreamCounter.count3(list, "hello"));

        /* La fonction list2 contient une liste de string représentant les nombres entier aléatoires générés.*/
        List<String> list2 =
            new Random(0)
                .ints(1_000_000, 0, 100)
                .mapToObj(Integer::toString)
                .collect(Collectors.toList());
    }
}
