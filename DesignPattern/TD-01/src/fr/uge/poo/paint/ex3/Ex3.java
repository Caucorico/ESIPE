package fr.uge.poo.paint.ex3;

import fr.uge.poo.simplegraphics.SimpleGraphics;

import java.awt.*;
import java.io.IOException;

public class Ex3 {

    public static void main(String[] args) throws IOException {

        if ( args.length != 1 ) {
            throw new IllegalArgumentException("Usage : Ex3 <filename>");
        }

        var figureList = SimpleGraphicsFileReader.readFile(args[0]);

        SimpleGraphics area = new SimpleGraphics("area", 800, 600);
        area.clear(Color.WHITE);
        area.render((graphics) -> {
            graphics.setColor(Color.BLACK);
            figureList.forEach( figure -> {
                figure.draw(graphics);
            });
        });
    }

}
