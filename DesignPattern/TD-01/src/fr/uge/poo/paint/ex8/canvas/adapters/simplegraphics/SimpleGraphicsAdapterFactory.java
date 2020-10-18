package fr.uge.poo.paint.ex8.canvas.adapters.simplegraphics;

import fr.uge.poo.paint.ex8.canvas.Canvas;
import fr.uge.poo.paint.ex8.canvas.CanvasFactory;
import fr.uge.poo.paint.ex8.canvas.CanvasInformations;
import fr.uge.poo.simplegraphics.SimpleGraphics;

public class SimpleGraphicsAdapterFactory implements CanvasFactory {

    @Override
    public Canvas withCanvasInformations(CanvasInformations canvasInformations) {
        SimpleGraphics area = new SimpleGraphics(canvasInformations.title(), canvasInformations.width(), canvasInformations.height());
        return new SimpleGraphicsAdapter(area);
    }

}
