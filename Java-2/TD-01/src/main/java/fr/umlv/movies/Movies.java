package fr.umlv.movies;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Movies {

    /**
     * The path of the file to read with the reader.
     */
    private String path;

    private Path objectPath;

    public Movies(String path) {
        this.path = path;
    }

    /**
     * This function return the Path object for the specified path.
     * @param path The string path of the file.
     * @return Return the Path object of the specified path.
     */
    static private Path getPath(String path) {
        return Path.of(path);
    }

    /**
     * This function init the Reader with the Path.
     */
    public void init() {
        this.objectPath = getPath(this.path);
    }

    /**
     * This function display of the lines of the file.
     *
     */
    public void displayAllLines() throws IOException {
        try(var stream = Files.lines(this.objectPath)) {
            stream.forEach(System.out::println);
        }
    }

    static public Map<String, List<String>> actorsByMovie(Path path) throws IOException {
        try(var stream = Files.lines(path)) {
            return stream.map( line -> line.split(";"))
                .collect(Collectors.toUnmodifiableMap(
                        e -> e[0],
                        e -> Arrays.stream(e).skip(1).collect(Collectors.toList())
                ));
        }
    }

    public static long numberOfUniqueActors(Map<String, List<String>> movies) {
        return movies.values().stream()
            .flatMap(Collection::stream)
            .distinct()
            .count();
    }

    public static Map<String, Long> numberOfMoviesByActor(Map<String, List<String>> movies) {
        return movies.values().stream()
                .flatMap(Collection::stream)
                .collect(Collectors.groupingBy(
                        Function.identity(), Collectors.counting()
                ));
    }

    public static Optional<Map.Entry<String, Long>> actorInMostMovies(Map<String, Long> map)
    {
        return map.entrySet().stream().max(Comparator.comparing(Map.Entry::getValue));
    }

    /**
     * Main
     * @param args args.
     */
    public static void main(String[] args) throws IOException {
        var path = Path.of("movies.txt");
        var actorsByMovie = Movies.actorsByMovie(path);
        var numberOfMoviesByActor = Movies.numberOfMoviesByActor(actorsByMovie);

        System.out.println(numberOfMoviesByActor.toString());
    }


}
