package fr.uge.poo.paint.rapport.canvas.adapters.simplegraphics;

import fr.uge.poo.paint.rapport.canvas.Canvas;
import fr.uge.poo.paint.rapport.canvas.CanvasFactory;
import fr.uge.poo.paint.rapport.canvas.CanvasInformations;
import fr.uge.poo.simplegraphics.SimpleGraphics;

public class SimpleGraphicsAdapterFactory implements CanvasFactory {

    @Override
    public Canvas withCanvasInformations(CanvasInformations canvasInformations) {
        SimpleGraphics area = new SimpleGraphics(canvasInformations.title(), canvasInformations.width(), canvasInformations.height());
        return new SimpleGraphicsAdapter(area);
    }

}
