package fr.uge.poo.paint.rapport;

import fr.uge.poo.paint.rapport.canvas.Canvas;
import fr.uge.poo.paint.rapport.canvas.CanvasFactory;
import fr.uge.poo.paint.rapport.canvas.CanvasInformations;
import fr.uge.poo.paint.rapport.canvas.adapters.simplegraphics.SimpleGraphicsAdapter;
import fr.uge.poo.simplegraphics.SimpleGraphics;

import java.io.IOException;
import java.util.ServiceLoader;

public class Rapport {

    public static void main(String[] args) throws IOException {

        String filename;

        if ( args.length == 1 ) {
            filename = args[0];
        }  else {
            throw new IllegalArgumentException("Usage : Rapport <filename>");
        }

        var figureSet = FigureSet.fromFile(filename);
        var width = figureSet.getMinWidth();
        var height = figureSet.getMinHeight();

        var canvasInformations = new CanvasInformations(Math.max(500, width), Math.max(500, height), "area");
        Canvas canvas;
        ServiceLoader<CanvasFactory> loader = ServiceLoader.load(CanvasFactory.class);

        var first = loader.findFirst();
        if ( first.isPresent() ) {
            canvas = first.get().withCanvasInformations(canvasInformations);
        } else {
            var simpleGraphics = new SimpleGraphics(canvasInformations.title(), canvasInformations.width(), canvasInformations.height());
            canvas = new SimpleGraphicsAdapter(simpleGraphics);
        }

        figureSet.draw(canvas);
        canvas.waitOnClick( (x, y) -> figureSet.displayNearestFigureColorized(x, y, canvas) );
    }

}
