package fr.uge.poo.paint.ex4;

import fr.uge.poo.paint.ex4.listener.ClickListener;
import fr.uge.poo.simplegraphics.SimpleGraphics;

import java.awt.*;
import java.io.IOException;

public class Ex4 {

    public static void main(String[] args) throws IOException {

        if ( args.length != 1 ) {
            throw new IllegalArgumentException("Usage : Ex4 <filename>");
        }

        SimpleGraphics area = new SimpleGraphics("area", 800, 600);
        var clickListener = new ClickListener(area);
        var figureSet = FigureSet.fromFile(args[0]);
        figureSet.draw(area);
        clickListener.register(figureSet::displayNearestFigureInTerminal);
    }

}
