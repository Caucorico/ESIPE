package fr.uge.poo.paint.ex4;

import java.awt.*;

public interface Figure {

    void draw(Graphics2D graphics);

    double distanceFromPointSquared(int x, int y);
}