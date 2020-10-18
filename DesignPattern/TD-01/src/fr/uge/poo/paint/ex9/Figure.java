package fr.uge.poo.paint.ex9;

import fr.uge.poo.paint.ex9.canvas.Canvas;

public interface Figure {

    void draw(Canvas canvas, Canvas.Color color);

    double distanceFromPointSquared(int x, int y);

    int getMaxHeight();

    int getMaxWidth();
}