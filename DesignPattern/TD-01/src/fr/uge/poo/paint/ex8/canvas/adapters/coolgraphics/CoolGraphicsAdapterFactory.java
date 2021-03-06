package fr.uge.poo.paint.ex8.canvas.adapters.coolgraphics;

import com.evilcorp.coolgraphics.CoolGraphics;
import fr.uge.poo.paint.ex8.canvas.Canvas;
import fr.uge.poo.paint.ex8.canvas.CanvasFactory;
import fr.uge.poo.paint.ex8.canvas.CanvasInformations;

public class CoolGraphicsAdapterFactory implements CanvasFactory {

    @Override
    public Canvas withCanvasInformations(CanvasInformations canvasInformations) {
        CoolGraphics area = new CoolGraphics(canvasInformations.title(), canvasInformations.width(), canvasInformations.height());
        return new CoolGraphicsAdapter(area);
    }

}
