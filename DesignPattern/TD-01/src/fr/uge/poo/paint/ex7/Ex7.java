package fr.uge.poo.paint.ex7;

import com.evilcorp.coolgraphics.CoolGraphics;
import fr.uge.poo.paint.ex7.canvas.Canvas;
import fr.uge.poo.paint.ex7.canvas.CoolGraphicsAdapter;
import fr.uge.poo.paint.ex7.canvas.SimpleGraphicsAdapter;
import fr.uge.poo.simplegraphics.SimpleGraphics;

import java.io.IOException;

public class Ex7 {

    public static void main(String[] args) throws IOException {

        boolean legacy = false;

        if ( args.length != 1 ) {
            throw new IllegalArgumentException("Usage : Ex7 <-legacy> <filename>");
        }

        var figureSet = FigureSet.fromFile(args[0]);
        var width = figureSet.getMinWidth();
        var height = figureSet.getMinHeight();

        Canvas canvas;
        if ( legacy ) {
            SimpleGraphics area = new SimpleGraphics("area", Math.max(500, width), Math.max(500, height));
            canvas = new SimpleGraphicsAdapter(area);
        } else {
            CoolGraphics area = new CoolGraphics("Example",Math.max(500, width),Math.max(500, height));
            canvas = new CoolGraphicsAdapter(area);
        }

        figureSet.draw(canvas);
        canvas.waitOnClick( (x, y) -> figureSet.displayNearestFigureColorized(x, y, canvas) );
    }

}
