package fr.uge.poo.simplegraphics;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.EventQueue;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;
import java.lang.reflect.InvocationTargetException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.function.Consumer;

import javax.swing.JButton;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.SwingUtilities;

/**
 *  This class represents a drawing area.
 *
 *  The method {@link #clear(Color)} can be used to set a background color.
 *  The method {@link #render(Consumer)} can be used to draw something one screen.
 *
 *  The mouse events can be retrieved using {@link #waitForMouseEvents(MouseCallback)}.
 *
 *  All these methods are thread safe.
 */
public class SimpleGraphics {
    private final JComponent area;
    private final BufferedImage buffer;
    final LinkedBlockingQueue<MouseEvent> eventBlockingQueue =
            new LinkedBlockingQueue<>();

    /**
     * Create a window of size width x height and a title.
     *
     * @param title the title of the window.
     * @param width the width of the window.
     * @param height the height of the window.
     */
    public SimpleGraphics(String title, int width, int height) {
        // This should be a factory method and not a constructor
        // but given that this library can be used
        // before static factory have been introduced
        // it's simpler from the user point of view
        // to create a canvas area using new.

        BufferedImage buffer = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        @SuppressWarnings("serial")
        JComponent area = new JComponent() {
            @Override
            protected void paintComponent(Graphics g) {
                g.drawImage(buffer, 0, 0, null);
            }

            @Override
            public Dimension getPreferredSize() {
                return new Dimension(width, height);
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
     * Clear the drawing area with the color of the brush.
     */
    public void clear(Color color) {
        render(graphics -> {
            graphics.setColor(color);
            graphics.fillRect(0, 0, area.getWidth(), area.getHeight());
        });
    }

    /**
     * Ask to render something on the screen.
     * @param consumer a code that will draw on the screen.
     */
    public void render(Consumer<Graphics2D> consumer) {
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
     *  @see SimpleGraphics#waitForMouseEvents(MouseCallback)
     */
    @FunctionalInterface
    public interface MouseCallback {
        /**
         * Called when the mouse is used inside the canvas.
         * @param x x coordinate of the mouse.
         * @param y y coordinate of the mouse
         *
         * @see SimpleGraphics#waitForMouseEvents(MouseCallback)
         */
        public void mouseClicked(int x, int y);
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
            mouseCallback.mouseClicked(mouseEvent.getX(), mouseEvent.getY());
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
