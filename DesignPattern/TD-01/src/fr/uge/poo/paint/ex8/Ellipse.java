package fr.uge.poo.paint.ex8;

import fr.uge.poo.paint.ex8.canvas.Canvas;

public class Ellipse extends Rectangulable {

    public Ellipse(int x1, int y1, int length, int height) {
        super(x1, y1, length, height);
    }

    @Override
    public void draw(Canvas canvas, Canvas.Color color) {
        canvas.drawEllipse(this.x1, this.y1, this.length, this.height, color);
    }

}
