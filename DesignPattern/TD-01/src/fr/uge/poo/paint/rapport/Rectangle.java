package fr.uge.poo.paint.rapport;

import fr.uge.poo.paint.rapport.canvas.Canvas;

public class Rectangle extends Rectangulable {

    public Rectangle(int x1, int y1, int length, int height) {
        super(x1, y1, length, height);
    }

    @Override
    public void draw(Canvas canvas, Canvas.Color color) {
        canvas.drawRectangle(this.x1, this.y1, this.length, this.height, color);
    }

    @Override
    public String toString() {
        return "Rectangle " + x1 + " " + y1 + " " + length + " " + height;
    }

    int getX() {
        return x1;
    }

    int getY() {
        return y1;
    }

    int getHeight() {
        return height;
    }
}
