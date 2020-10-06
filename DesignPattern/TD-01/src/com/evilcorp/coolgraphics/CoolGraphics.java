package com.evilcorp.coolgraphics;


import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;
import java.lang.reflect.InvocationTargetException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.function.Consumer;

/**
 *  This class represents a drawing area with a nice red border.
 *
 *  The method {@link #repaint(ColorPlus)} (Color)} can be used to set a background color.
 *
 *
 *  The mouse events can be retrieved using {@link #waitForMouseEvents(MouseCallback)}.
 *
 *  All these methods are thread safe.
 */
public class CoolGraphics {
    private final JComponent area;
    private final BufferedImage buffer;
    private static final int border = 10;
    final LinkedBlockingQueue<MouseEvent> eventBlockingQueue =
            new LinkedBlockingQueue<>();

    public enum ColorPlus {
        RED(Color.RED),
        GREEN(Color.GREEN),
        ORANGE(Color.ORANGE),
        BLUE(Color.BLUE),
        BLACK(Color.BLACK),
        WHITE(Color.WHITE);

        private final Color color;

        ColorPlus(Color color) {
            this.color = color;
        }
    }

    /**
     * Create a window of size width x height and a title.
     *
     * @param title the title of the window.
     * @param width the width of the window.
     * @param height the height of the window.
     */
    public CoolGraphics(String title, int width, int height) {
        // This should be a factory method and not a constructor
        // but given that this library can be used
        // before static factory have been introduced
        // it's simpler from the user point of view
        // to create a canvas area using new.



        BufferedImage buffer = new BufferedImage(width+2*border, height+2*border, BufferedImage.TYPE_INT_ARGB);
        @SuppressWarnings("serial")
        JComponent area = new JComponent() {
            @Override
            protected void paintComponent(Graphics g) {
                g.drawImage(buffer, 0, 0, null);
            }

            @Override
            public Dimension getPreferredSize() {
                return new Dimension(width+2*border, height+2*border);
            }
        };
        class MouseManager extends MouseAdapter implements MouseMotionListener {
            @Override
            public void mouseClicked(MouseEvent event) {
                try {
                    eventBlockingQueue.put(event);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            @Override
            public void mouseMoved(MouseEvent e) {
                // do nothing
            }
            @Override
            public void mouseDragged(MouseEvent e) {
                mouseClicked(e);
            }
        }
        MouseManager mouseManager = new MouseManager();
        area.addMouseListener(mouseManager);
        area.addMouseMotionListener(mouseManager);
        try {
            EventQueue.invokeAndWait(() -> {
                JFrame frame = new JFrame(title);
                area.setOpaque(true);
                frame.setContentPane(area);
                frame.setResizable(false);
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.pack();
                frame.setVisible(true);
            });
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } catch(InvocationTargetException e) {
            throw new IllegalStateException(e.getCause());
        }
        this.buffer = buffer;
        this.area = area;
    }


    /**
     * Repaint the drawing area with the color of the brush.
     */
    public void repaint(ColorPlus color) {
        render(graphics -> {
            graphics.setColor(Color.RED);
            graphics.fillRect(0, 0, area.getWidth(), area.getHeight());
            graphics.setColor(color.color);
            graphics.fillRect(border, border, area.getWidth()-2*border, area.getHeight()-2*border);
        });
    }

    /**
     * Draw a line between (x1,y1) and (x2,y2) with the given color
     * @param x1 the x coordinate of the first point of the line
     * @param y1 the y coordinate of the first point of the line
     * @param x2 the x coordinate of the second point of the line
     * @param y2 the y coordinate of the second point of the line
     * @param color color of the line
     */
    public void drawLine(int x1,int y1,int x2,int y2, ColorPlus color){
        render( graphics -> {
            graphics.setColor(color.color);
            graphics.drawLine(border+x1,border+y1,border+x2,border+y2);
        });
    }

    /**
     * Draw a line between (x1,y1) and (x2,y2) with the given color
     * @param x,y the coordinate of the center of the ellipse
     * @param width of the ellipse
     * @param height of the ellipse
     * @param color the color of the ellipse
     */
    public void drawEllipse(int x,int y,int width,int height, ColorPlus color){
        render( graphics -> {
            graphics.setColor(color.color);
            graphics.drawOval(x+border,y+border,width,height);
        });
    }

    /**
     * Ask to render something on the screen.
     * @param consumer a code that will draw on the screen.
     */
    private void render(Consumer<Graphics2D> consumer) {
        EventQueue.invokeLater(() -> {
            Graphics2D graphics = buffer.createGraphics();
            try {
                graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                consumer.accept(graphics);
                area.repaint();  // repost a paint event to avoid blinking
            } finally {
                graphics.dispose();
            }
        });
    }

    /**
     *  A functional interface of the mouse callback.
     *  @see CoolGraphics#waitForMouseEvents(MouseCallback)
     */
    @FunctionalInterface
    public interface MouseCallback {
        /**
         * Called when the mouse is used inside the canvas.
         * @param x x coordinate of the mouse.
         * @param y y coordinate of the mouse
         *
         * @see CoolGraphics#waitForMouseEvents(MouseCallback)
         */
        void mouseClicked(int x, int y);
    }

    /**
     * Wait for mouse events, the mouseCallback method
     * {@link MouseCallback#mouseClicked(int, int)}
     * will be called for each mouse event until the window
     * is {@link #close() closed}.
     *
     * @param mouseCallback a mouse callback.
     *
     * @throws IllegalStateException if this method is
     *         called by the event dispatch thread.
     */
    public void waitForMouseEvents(MouseCallback mouseCallback) {
        if (EventQueue.isDispatchThread()) {
            throw new IllegalStateException("This method can not be called from the EDT");
        }
        for(;;) {
            MouseEvent mouseEvent;
            try {
                mouseEvent = eventBlockingQueue.take();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
            if (mouseEvent == CLOSE_EVENT) {
                return;
            }
            if (mouseEvent.getX()<border || mouseEvent.getY()<border || mouseEvent.getX()>=area.getWidth()-2*border || mouseEvent.getY()>=area.getHeight()-2*border){
                continue;
            }
            mouseCallback.mouseClicked(mouseEvent.getX()-border, mouseEvent.getY()-border);
        }
    }

    private static final MouseEvent CLOSE_EVENT =
            new MouseEvent(new JButton(), -1, -1, -1, 0, 0, 0, 0, 0, false, 0);

    /**
     * Close the window.
     */
    public void close() {
        JFrame frame = (JFrame)SwingUtilities.getRoot(area);
        frame.dispose();
        try {
            eventBlockingQueue.put(CLOSE_EVENT);
        } catch(InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
