package fr.uge.poo.paint.ex7.canvas;

import fr.uge.poo.simplegraphics.SimpleGraphics;

import java.awt.*;

public class SimpleGraphicsAdapter implements Canvas {

    private final SimpleGraphics simpleGraphics;

    public SimpleGraphicsAdapter(SimpleGraphics simpleGraphics) {
        this.simpleGraphics = simpleGraphics;
    }

    private static java.awt.Color colorConvertor(Color color) {
        return switch (color) {
            case BLACK -> java.awt.Color.BLACK;
            case WHITE -> java.awt.Color.WHITE;
            case ORANGE -> java.awt.Color.ORANGE;
        };
    }

    @Override
    public void drawLine(int x1, int y1, int x2, int y2, Color color) {
        simpleGraphics.render((graphics2D -> {
            graphics2D.setColor(colorConvertor(color));
            graphics2D.drawLine(x1, y1, x2, y2);
        }));
    }


    @Override
    public void drawRectangle(int x1, int y1, int length, int height, Color color) {
        simpleGraphics.render((graphics2D -> {
            graphics2D.setColor(colorConvertor(color));
            graphics2D.drawRect(x1, y1, length, height);
        }));
    }


    @Override
    public void drawEllipse(int x1, int y1, int length, int height, Color color) {
        simpleGraphics.render((graphics2D -> {
            graphics2D.setColor(colorConvertor(color));
            graphics2D.drawOval(x1, y1, length, height);
        }));
    }

    @Override
    public void clear(Color color) {
        simpleGraphics.clear(colorConvertor(color));
    }

    @Override
    public void waitOnClick(MouseCallback callback) {
        simpleGraphics.waitForMouseEvents(callback::onMouseEvent);
    }
}
