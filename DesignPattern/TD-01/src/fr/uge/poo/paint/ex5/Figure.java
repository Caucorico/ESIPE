package fr.uge.poo.paint.ex5;

import java.awt.*;

public interface Figure {

    void draw(Graphics2D graphics);

    void draw(Graphics2D graphics, Color color);

    double distanceFromPointSquared(int x, int y);
}