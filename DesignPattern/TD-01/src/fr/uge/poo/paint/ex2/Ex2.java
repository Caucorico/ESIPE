package fr.uge.poo.paint.ex2;

import fr.uge.poo.simplegraphics.SimpleGraphics;
import fr.uge.poo.simplegraphics.SimpleGraphicsExample;

import java.awt.Graphics2D;
import java.awt.Color;
import java.io.IOException;
import java.util.List;

public class Ex2 {

    public static void main(String[] args) throws IOException {

        if ( args.length != 1 ) {
            throw new IllegalArgumentException("Usage : Ex2 <filename>");
        }

        var figureList = SimpleGraphicsFileReader.readFile(args[0]);

        SimpleGraphics area = new SimpleGraphics("area", 800, 600);
        area.clear(Color.WHITE);
        area.render((graphics) -> {
            figureList.forEach( figure -> {
                figure.draw(graphics);
            });
        });
    }

}
