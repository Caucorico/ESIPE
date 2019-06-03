package fr.umlv.tp10.exercice01;

import java.util.List;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class StreamCounter
{
    public static long count(List<String> list, String element)
    {
        /* Une interface List implements l'interface Colliction qui implément une méthode stream()  qui renvoie une stream. */
        Stream<String> s = list.stream();

        /* La méthode filter() de stream prend un Predicate en argument et applique ce filtre sur la Stream  */
        /* La fonction lambda correspondante s'ecrit w -> element.equals(w) et peut etre simplifié de cette manière : element::equals */
        s = s.filter(element::equals);

        /* Enfin une méthode count permet de compter le nombre d'element présent dans la Stream. */
        return s.count();
    }

    public static int count3(List<String> list, String element)
    {
        int counter = 0;
        Stream<String> s = list.stream();
        IntStream s2 = s.mapToInt(w -> element.equals(w)?1:0 );
        return s2.collect(w -> counter+w);
    }
}
