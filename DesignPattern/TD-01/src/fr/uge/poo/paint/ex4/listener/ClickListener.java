package fr.uge.poo.paint.ex4.listener;

import fr.uge.poo.simplegraphics.SimpleGraphics;

public class ClickListener {

    private final SimpleGraphics area;

    public ClickListener(SimpleGraphics area) {
        this.area = area;
    }

    public void register(SimpleGraphics.MouseCallback callback) {
        area.waitForMouseEvents(callback);
    }
}
