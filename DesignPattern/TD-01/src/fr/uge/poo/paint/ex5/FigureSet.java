package fr.uge.poo.paint.ex5;

import fr.uge.poo.simplegraphics.SimpleGraphics;

import java.awt.*;
import java.io.IOException;
import java.util.List;
import java.util.*;

public class FigureSet {

    private final HashMap<Integer, Figure> figures;

    public FigureSet() {
        figures = new HashMap<>();
    }

    public FigureSet(HashMap<Integer, Figure> figures) {
        Objects.requireNonNull(figures);
        this.figures = figures;
    }

    public FigureSet(List<Figure> figures) {
        this();
        Objects.requireNonNull(figures);
        figures.forEach((figure) -> this.figures.put(figure.hashCode(), figure));
    }

    public static FigureSet fromFile(String filename) throws IOException {
        return new FigureSet(SimpleGraphicsFileReader.readFile(filename));
    }

    public Optional<Figure> nearestOfPoint(int x, int y) {
        HashSet<Figure> hs = new HashSet<>();

        return figures.entrySet().stream().min( (a, b) -> {
            var distance = a.getValue().distanceFromPointSquared(x, y) - b.getValue().distanceFromPointSquared(x, y);

            if ( distance > 0.0D ) {
                return 1;
            } else if ( distance < 0.0D ) {
                return -1;
            }

            return 0;
        }).map(Map.Entry::getValue);
    }

    public void draw(SimpleGraphics area) {
        area.clear(Color.WHITE);
        area.render((graphics) -> {
            graphics.setColor(Color.BLACK);
            figures.forEach( (key, figure) -> figure.draw(graphics));
        });
    }

    public void displayNearestFigureInTerminal(int x, int y) {
        var optional = nearestOfPoint(x, y);

        if ( optional.isEmpty() ) {
            System.out.println("No figure exist in this set.");
        } else {
            System.out.println(optional.get().toString());
        }
    }

    public void displayNearestFigureColorized(int x, int y, SimpleGraphics area) {
        var optional = nearestOfPoint(x, y);

        draw(area);

        optional.ifPresent( (figure) -> area.render((graphics) -> figure.draw(graphics, Color.ORANGE)));
    }

}
