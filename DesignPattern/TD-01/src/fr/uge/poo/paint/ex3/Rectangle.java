package fr.uge.poo.paint.ex3;

import java.awt.*;

public class Rectangle extends Rectangulable {
    public Rectangle(int x1, int y1, int length, int height) {
        super(x1, y1, length, height);
    }

    @Override
    public void draw(Graphics2D graphics) {
        graphics.drawRect(this.x1, this.y1, this.length, this.height);
    }
}
