package fr.upem.net.udp.nonblocking;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.DatagramChannel;
import java.nio.channels.SelectionKey;
import java.nio.channels.Selector;
import java.util.ArrayList;
import java.util.logging.Logger;

public class ServerEchoMultiPort {

    private static class Context {

        private InetSocketAddress inetSocketAddress;

        private final ByteBuffer byteBuffer;

        private Context() {
            this.byteBuffer = ByteBuffer.allocateDirect(BUFFER_SIZE);
        }

        public InetSocketAddress getInetSocketAddress() {
            return inetSocketAddress;
        }

        public void setInetSocketAddress(InetSocketAddress inetSocketAddress) {
            this.inetSocketAddress = inetSocketAddress;
        }

        public ByteBuffer getByteBuffer() {
            return byteBuffer;
        }
    }

    private static final Logger logger = Logger.getLogger(ServerEchoPlus.class.getName());

    private final ArrayList<DatagramChannel> datagramChannels;

    private static final int BUFFER_SIZE = 1024;

    private final Selector selector;

    public ServerEchoMultiPort(int startPort, int endPort) throws IOException {
        var portNumber = endPort - (startPort-1);

        if ( portNumber < 0 ) {
            throw new IllegalArgumentException();
        }

        datagramChannels = new ArrayList<>(portNumber);
        selector = Selector.open();

        for ( var i = startPort ; i <= endPort  ; i++ ) {
            var inet = new InetSocketAddress(i);
            var dc = DatagramChannel.open();
            dc.bind(inet);
            dc.configureBlocking(false);
            dc.register(selector, SelectionKey.OP_READ, new Context());
            datagramChannels.add(dc);
        }
    }

    public void serve() throws IOException {
        logger.info("ServerEcho started");

        try {
            while (!Thread.interrupted()) {
                selector.select(this::treatKey);
            }
        } catch (UncheckedIOException e) {
            throw e.getCause();
        }
    }

    private void treatKey(SelectionKey key) {
        try{
            if (key.isValid() && key.isWritable()) {
                doWrite(key);
            }
            if (key.isValid() && key.isReadable()) {
                doRead(key);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }

    }

    private void doRead(SelectionKey key) throws IOException {
        Context context = (Context)key.attachment();
        ByteBuffer byteBuffer = context.getByteBuffer();
        DatagramChannel datagramChannel = (DatagramChannel) key.channel();
        byteBuffer.clear();

        InetSocketAddress exp = (InetSocketAddress) datagramChannel.receive(byteBuffer);
        if (  exp == null ) {
            return;
        }

        byteBuffer.flip();
        context.setInetSocketAddress(exp);
        key.interestOps(SelectionKey.OP_WRITE);
    }

    private void doWrite(SelectionKey key) throws IOException {
        Context context = (Context)key.attachment();
        ByteBuffer byteBuffer = context.getByteBuffer();
        DatagramChannel datagramChannel = (DatagramChannel) key.channel();

        datagramChannel.send(byteBuffer, context.getInetSocketAddress());
        if ( byteBuffer.hasRemaining() ) {
            return;
        }

        key.interestOps(SelectionKey.OP_READ);
    }

    public static void usage() {
        System.out.println("Usage : ServerEcho portStart portEnd");
    }

    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            usage();
            return;
        }
        ServerEchoMultiPort server= new ServerEchoMultiPort(Integer.valueOf(args[0]), Integer.valueOf(args[1]));
        server.serve();
    }
}
