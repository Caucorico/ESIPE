package fr.uge.poo.paint.ex4;

import java.awt.*;

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

    @Override
    public void draw(Graphics2D graphics) {
        graphics.drawLine(this.x1, this.y1, this.x2, this.y2);
    }
}
