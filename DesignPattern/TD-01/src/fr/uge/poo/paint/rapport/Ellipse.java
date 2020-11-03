package fr.uge.poo.paint.rapport;

import fr.uge.poo.paint.rapport.canvas.Canvas;

public class Ellipse extends Rectangulable {

    public Ellipse(int x1, int y1, int length, int height) {
        super(x1, y1, length, height);
    }

    @Override
    public void draw(Canvas canvas, Canvas.Color color) {
        canvas.drawEllipse(this.x1, this.y1, this.length, this.height, color);
    }

    @Override
    public String toString() {
        return "Ellipse " + x1 + " " + y1 + " " + length + " " + height;
    }
}
