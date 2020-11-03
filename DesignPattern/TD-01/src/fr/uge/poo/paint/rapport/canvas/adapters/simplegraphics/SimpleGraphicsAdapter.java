package fr.uge.poo.paint.rapport.canvas.adapters.simplegraphics;

import fr.uge.poo.paint.rapport.canvas.Canvas;
import fr.uge.poo.simplegraphics.SimpleGraphics;

import java.awt.*;
import java.util.ArrayDeque;
import java.util.Queue;
import java.util.function.Consumer;

public class SimpleGraphicsAdapter implements Canvas {

    private final SimpleGraphics simpleGraphics;

    private Consumer<Graphics2D> consumer;

    public SimpleGraphicsAdapter(SimpleGraphics simpleGraphics) {
        this.simpleGraphics = simpleGraphics;
    }

    private static java.awt.Color colorConvertor(Color color) {
        return switch (color) {
            case BLACK -> java.awt.Color.BLACK;
            case WHITE -> java.awt.Color.WHITE;
            case ORANGE -> java.awt.Color.ORANGE;
        };
    }

    private void addLambda(Consumer<Graphics2D> consumer) {
        if ( this.consumer == null ) {
            this.consumer = consumer;
        } else {
            this.consumer = this.consumer.andThen(consumer);
        }
    }

    @Override
    public void drawLine(int x1, int y1, int x2, int y2, Color color) {
        addLambda( graphics2D -> {
            graphics2D.setColor(colorConvertor(color));
            graphics2D.drawLine(x1, y1, x2, y2);
        });
    }


    @Override
    public void drawRectangle(int x1, int y1, int length, int height, Color color) {
        addLambda(graphics2D -> {
            graphics2D.setColor(colorConvertor(color));
            graphics2D.drawRect(x1, y1, length, height);
        });
    }


    @Override
    public void drawEllipse(int x1, int y1, int length, int height, Color color) {
        addLambda(graphics2D -> {
            graphics2D.setColor(colorConvertor(color));
            graphics2D.drawOval(x1, y1, length, height);
        });
    }

    @Override
    public void clear(Color color) {
        simpleGraphics.clear(colorConvertor(color));
    }

    @Override
    public void waitOnClick(MouseCallback callback) {
        simpleGraphics.waitForMouseEvents(callback::onMouseEvent);
    }

    @Override
    public void render() {
        simpleGraphics.render( graphics2D -> {
            if ( consumer != null ) {
                consumer.accept(graphics2D);
            }
            consumer = null;

            /* ################################################### */

            /*while (!queue.isEmpty()) {
                var process = queue.poll();
                process.accept(graphics2D);
            }*/

            /* ################################################### */

            /*var iterator = queue.iterator();
            System.out.println(iterator.hasNext()+ "");
            while (iterator.hasNext()) {
                var n = iterator.next();
                n.accept(graphics2D);
            }
            queue.clear();*/

            /* #################################################### */

            /*queue.forEach( process -> process.accept(graphics2D));
            queue.clear();*/
        });
    }
}
