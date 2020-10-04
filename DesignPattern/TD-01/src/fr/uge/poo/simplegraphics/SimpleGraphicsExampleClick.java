package fr.uge.poo.simplegraphics;

import java.awt.Color;

public class SimpleGraphicsExampleClick {

    private static void callback(SimpleGraphics area, int x, int y) {
        area.render(graphics -> {
            graphics.setColor(Color.BLACK);
            graphics.drawRect(x - 18, y - 18, 36, 36);
        });
    }

    public static void main(String[] args) {
        var area = new SimpleGraphics("area", 800, 600);
        area.clear(Color.WHITE);
        area.waitForMouseEvents((x, y) -> callback(area, x, y));
    }
}
