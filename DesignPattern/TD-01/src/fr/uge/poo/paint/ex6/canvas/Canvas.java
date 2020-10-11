package fr.uge.poo.paint.ex6.canvas;

public interface Canvas {

    enum Color {
        BLACK, WHITE, ORANGE
    }

    @FunctionalInterface
    interface MouseCallback {
        void onMouseEvent(int x, int y);
    }

    void drawLine(int x1, int y1, int x2, int y2, Color color);

    void drawRectangle(int x1, int y1, int length, int height, Color color);

    void drawEllipse(int x1, int y1, int length, int height, Color color);

    void clear(Color color);

    void waitOnClick(MouseCallback callback);
}
