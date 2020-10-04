package fr.uge.poo.paint.ex5;

import fr.uge.poo.paint.ex5.listener.ClickListener;
import fr.uge.poo.simplegraphics.SimpleGraphics;

import java.io.IOException;

public class Ex5 {

    public static void main(String[] args) throws IOException {

        if ( args.length != 1 ) {
            throw new IllegalArgumentException("Usage : Ex5 <filename>");
        }

        SimpleGraphics area = new SimpleGraphics("area", 800, 600);
        var clickListener = new ClickListener(area);
        var figureSet = FigureSet.fromFile(args[0]);
        figureSet.draw(area);
        clickListener.register( (x, y) -> figureSet.displayNearestFigureColorized(x, y, area));
    }

}
