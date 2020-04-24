package fr.upem.net.tcp;

import java.io.IOException;
import java.nio.channels.SocketChannel;

public class ThreadData {

    private final Object lockClientConnectedSocketChannel = new Object();
    private SocketChannel clientConnectedSocketChannel;

    private final Object lockLastActivity = new Object();
    private long lastActivity;

    public ThreadData setSocketChannel(SocketChannel clientConnectedSocketChannel) {
        synchronized (lockClientConnectedSocketChannel) {
            this.clientConnectedSocketChannel = clientConnectedSocketChannel;
        }
        tick();
        return this;
    }

    public SocketChannel getSocketChannel() {
        synchronized (lockClientConnectedSocketChannel) {
            return this.clientConnectedSocketChannel;
        }
    }

    public void tick() {
        synchronized (lockLastActivity) {
            lastActivity = System.currentTimeMillis();
        }
    }

    public void close() throws IOException {
        synchronized (lockClientConnectedSocketChannel) {
            if (clientConnectedSocketChannel != null ) {
                /* TODO : Ask to the teacher if close input first? */

                clientConnectedSocketChannel.close();
                clientConnectedSocketChannel = null;
            }
        }
    }

    public void closeIfInactive(int timeout) throws IOException {
        if ( lastActivity + timeout < System.currentTimeMillis() ) {
            close();
        }
    }

    public boolean isActive() {
        synchronized (lockClientConnectedSocketChannel) {
            return clientConnectedSocketChannel != null;
        }
    }
}
