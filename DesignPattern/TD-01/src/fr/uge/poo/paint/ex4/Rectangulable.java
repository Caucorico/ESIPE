package fr.uge.poo.paint.ex4;

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
}
