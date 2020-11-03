package fr.uge.poo.paint.rapport.canvas.adapters.coolgraphics;

import com.evilcorp.coolgraphics.CoolGraphics;
import fr.uge.poo.paint.rapport.canvas.Canvas;
import fr.uge.poo.paint.rapport.canvas.CanvasFactory;
import fr.uge.poo.paint.rapport.canvas.CanvasInformations;

public class CoolGraphicsAdapterFactory implements CanvasFactory {

    @Override
    public Canvas withCanvasInformations(CanvasInformations canvasInformations) {
        CoolGraphics area = new CoolGraphics(canvasInformations.title(), canvasInformations.width(), canvasInformations.height());
        return new CoolGraphicsAdapter(area);
    }

}
