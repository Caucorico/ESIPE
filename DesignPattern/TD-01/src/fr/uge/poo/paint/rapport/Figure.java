package fr.uge.poo.paint.rapport;

import fr.uge.poo.paint.rapport.canvas.Canvas;

public interface Figure {

    void draw(Canvas canvas, Canvas.Color color);

    double distanceFromPointSquared(int x, int y);

    int getMaxHeight();

    int getMaxWidth();
}