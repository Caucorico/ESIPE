package fr.uge.poo.paint.ex7;

import fr.uge.poo.paint.ex7.canvas.Canvas;

public class Line implements Figure {

    private final int x1;
    private final int x2;
    private final int y1;
    private final int y2;

    public Line(int x1, int y1, int x2, int y2) {
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
    }

    private double xCenter() {
        return (x1 + x2)/2.0;
    }

    private double yCenter() {
        return (y1 + y2)/2.0;
    }

    @Override
    public void draw(Canvas canvas, Canvas.Color color) {
        canvas.drawLine(this.x1, this.y1, this.x2, this.y2, color);
    }

    @Override
    public double distanceFromPointSquared(int x, int y) {
        return Math.pow(x - xCenter(), 2) + Math.pow(y - yCenter(), 2);
    }
}
