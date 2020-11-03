package fr.uge.poo.paint.rapport;

import fr.uge.poo.paint.rapport.canvas.Canvas;

public class Square implements Figure {

    private final Rectangle rectangle;

    public Square(int x, int y, int sideSize) {
        this.rectangle = new Rectangle(x, y, sideSize, sideSize);
    }

    @Override
    public void draw(Canvas canvas, Canvas.Color color) {
        rectangle.draw(canvas, color);
    }

    @Override
    public double distanceFromPointSquared(int x, int y) {
        return rectangle.distanceFromPointSquared(x, y);
    }

    @Override
    public int getMaxHeight() {
        return rectangle.getMaxHeight();
    }

    @Override
    public int getMaxWidth() {
        return rectangle.getMaxWidth();
    }

    @Override
    public String toString() {
        return "Square " + rectangle.getX() + " " + rectangle.getY() + " " + rectangle.getHeight() + " " + rectangle.getHeight();
    }

}
