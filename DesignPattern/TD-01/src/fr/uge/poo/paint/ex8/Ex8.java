package fr.uge.poo.paint.ex8;

import fr.uge.poo.paint.ex8.canvas.*;
import fr.uge.poo.paint.ex8.canvas.adapters.simplegraphics.SimpleGraphicsAdapter;
import fr.uge.poo.simplegraphics.SimpleGraphics;

import java.io.IOException;
import java.util.ServiceLoader;

public class Ex8 {

    public static void main(String[] args) throws IOException {

        String filename;

        if ( args.length == 1 ) {
            filename = args[0];
        }  else {
            throw new IllegalArgumentException("Usage : Ex8 <filename>");
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
