package fr.uge.poo.paint.ex6;

import fr.uge.poo.paint.ex6.canvas.Canvas;

import java.io.IOException;
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

    public void draw(Canvas canvas) {
        canvas.clear(Canvas.Color.WHITE);
        figures.forEach( (key, figure) -> figure.draw(canvas, Canvas.Color.BLACK) );
    }

    public void displayNearestFigureInTerminal(int x, int y) {
        var optional = nearestOfPoint(x, y);

        if ( optional.isEmpty() ) {
            System.out.println("No figure exist in this set.");
        } else {
            System.out.println(optional.get().toString());
        }
    }

    public void displayNearestFigureColorized(int x, int y, Canvas canvas) {
        var optional = nearestOfPoint(x, y);

        draw(canvas);

        optional.ifPresent( (figure) -> figure.draw(canvas, Canvas.Color.ORANGE));
    }

    public int getMinHeight() {
        var optional = figures.values().stream().map(Figure::getMaxHeight).max(Integer::compareTo);
        if (optional.isEmpty()) {
            return 0;
        }

        return optional.get();
    }

    public int getMinWidth() {
        var optional = figures.values().stream().map(Figure::getMaxWidth).max(Integer::compareTo);
        if (optional.isEmpty()) {
            return 0;
        }

        return optional.get();
    }

}
