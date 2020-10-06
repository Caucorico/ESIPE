package fr.uge.poo.paint.ex7;import java.awt.*;


abstract class Rectangulable implements Figure {

    protected final int x1;
    protected final int y1;
    protected final int length;
    protected final int height;

    public Rectangulable(int x1, int y1, int length, int height) {
        this.x1 = x1;
        this.y1 = y1;
        this.length = length;
        this.height = height;
    }

    private double xCenter() {
        return x1 + (length/2.0);
    }

    private double yCenter() {
        return y1 + (height/2.0);
    }

    @Override
    public double distanceFromPointSquared(int x, int y) {
        return Math.pow(x - xCenter(), 2) + Math.pow(y - yCenter(), 2);
    }
}
