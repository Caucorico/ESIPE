package fr.uge.poo.paint.ex5;

import java.awt.*;

public class Ellipse extends Rectangulable {

    public Ellipse(int x1, int y1, int length, int height) {
        super(x1, y1, length, height);
    }

    @Override
    public void draw(Graphics2D graphics) {
        graphics.drawOval(this.x1, this.y1, this.length, this.height);
    }

    @Override
    public void draw(Graphics2D graphics, Color color) {
        graphics.setColor(color);
        this.draw(graphics);
    }

}
