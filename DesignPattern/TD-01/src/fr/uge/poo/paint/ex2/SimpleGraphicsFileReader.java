package fr.uge.poo.paint.ex2;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

public class SimpleGraphicsFileReader {

    public static List<Figure> readFile(String filename) throws IOException {
        Path path = Paths.get(filename);
        ArrayList<Figure> figures = new ArrayList<>();

        try(Stream<String> lines = Files.lines(path)) {
            lines.forEach(line -> {
                var split = line.split(" ");

                switch (split[0]) {
                    case "line":
                        figures.add(
                                new Line(Integer.parseInt(split[1]),
                                        Integer.parseInt(split[2]),
                                        Integer.parseInt(split[3]),
                                        Integer.parseInt(split[4]))
                        );
                        break;
                    default:
                        throw new UnsupportedOperationException("TODO STORE");
                }

            });
        }

        return figures;
    }

}
