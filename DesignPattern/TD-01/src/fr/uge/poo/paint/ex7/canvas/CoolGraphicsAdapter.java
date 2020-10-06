package fr.uge.poo.paint.ex7.canvas;

import com.evilcorp.coolgraphics.CoolGraphics;

public class CoolGraphicsAdapter implements Canvas {

    private final CoolGraphics coolGraphics;

    public CoolGraphicsAdapter(CoolGraphics coolGraphics) {
        this.coolGraphics = coolGraphics;
    }

    private static CoolGraphics.ColorPlus colorConvertor(Color color) {
        return switch (color) {
            case BLACK -> CoolGraphics.ColorPlus.BLACK;
            case WHITE -> CoolGraphics.ColorPlus.WHITE;
            case ORANGE -> CoolGraphics.ColorPlus.ORANGE;
        };
    }

    @Override
    public void drawLine(int x1, int y1, int x2, int y2, Color color) {
        coolGraphics.drawLine(x1, y1, x2, y2, colorConvertor(color));
    }

    @Override
    public void drawRectangle(int x1, int y1, int length, int height, Color color) {
        coolGraphics.drawLine(x1, y1, x1+length, y1, colorConvertor(color));
        coolGraphics.drawLine(x1, y1, x1, y1+height, colorConvertor(color));
        coolGraphics.drawLine(x1, y1+height, x1+length, y1+height, colorConvertor(color));
        coolGraphics.drawLine(x1+length, y1, x1+length, y1+height, colorConvertor(color));
    }

    @Override
    public void drawEllipse(int x1, int y1, int length, int height, Color color) {
        coolGraphics.drawEllipse(x1, y1, length, height, colorConvertor(color));
    }

    @Override
    public void clear(Color color) {
        coolGraphics.repaint(colorConvertor(color));
    }

    @Override
    public void waitOnClick(MouseCallback callback) {
        coolGraphics.waitForMouseEvents(callback::onMouseEvent);
    }
}
