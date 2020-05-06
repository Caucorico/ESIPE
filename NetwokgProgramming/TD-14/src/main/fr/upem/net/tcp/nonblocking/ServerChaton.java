package fr.upem.net.tcp.nonblocking;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.*;
import java.util.LinkedList;
import java.util.Queue;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ServerChaton {

    static private int BUFFER_SIZE = 2_048;
    static private Logger logger = Logger.getLogger(ServerChatInt.class.getName());
    final private ServerSocketChannel serverSocketChannel;
    private final Selector selector;

    private static class Context {
        final private SocketChannel sc;
        final private ServerChaton server;
        final private SelectionKey key;
        final private ByteBuffer bbin = ByteBuffer.allocate(BUFFER_SIZE);
        final private ByteBuffer bbout = ByteBuffer.allocate(BUFFER_SIZE);
        final private Queue<Message> queue = new LinkedList<>();
        private boolean closed = false;
        final private MessageReader messageReader = new MessageReader();

        public Context(SocketChannel sc, ServerChaton server, SelectionKey key) {
            this.sc = sc;
            this.server = server;
            this.key = key;
        }

        private void queueMessage(Message msg) {
            queue.add(msg);
            processOut();
        }

        private void silentlyClose() {
            try {
                sc.close();
            } catch (IOException e) {
                // ignore exception
            }
        }

        private void updateInterestOps() {
            byte interestOps = 0x0;

            if ( !closed && bbin.hasRemaining() ) {
                interestOps |= SelectionKey.OP_READ;
            }

            if ( bbout.position() > 0 ) {
                interestOps |= SelectionKey.OP_WRITE;
            }

            if ( interestOps == 0x0 ) {
                silentlyClose();
                return;
            }

            key.interestOps(interestOps);
        }

        private void processIn() {
            Reader.ProcessStatus status;
            do {
                status = messageReader.process(bbin);
                switch (status) {
                    case DONE:
                        server.broadcast(messageReader.get());
                        messageReader.reset();
                        break;
                    case REFILL:
                        break;
                    case ERROR:
                        closed = true;
                        logger.log(Level.WARNING, "Error durung reading !!!");
                        break;
                }
            } while( status == Reader.ProcessStatus.REFILL );

            updateInterestOps();
        }

        private void processOut() {
            /* This is absolutly not optimized */
            if ( bbout.position() == 0 && !queue.isEmpty()) {
                bbout.put(queue.remove().getBufferized().flip());
            }

            updateInterestOps();
        }

        public void doRead() throws IOException {
            if ( !bbin.hasRemaining() ) {
                logger.warning("Call doRead but the buffer is full !");
            }

            var res = sc.read(bbin);
            if ( res == -1 ) {
                closed = true;
                updateInterestOps();
                return;
            }

            processIn();
        }

        public void doWrite() throws  IOException {
            if ( bbout.position() == 0 ) {
                logger.warning("Call doWrite but the buffer is empty !");
            }

            bbout.flip();
            sc.write(bbout);
            bbout.compact();

            processOut();
            //updateInterestOps();
        }

    }

    public ServerChaton(int port) throws IOException {
        serverSocketChannel = ServerSocketChannel.open();
        serverSocketChannel.bind(new InetSocketAddress(port));
        selector = Selector.open();
    }

    public void broadcast(Message message) {
        selector.keys().stream().filter(key -> key.interestOps() != SelectionKey.OP_ACCEPT).forEach(key -> {
            var context = (Context)key.attachment();
            context.queueMessage(message);
        });
    }

    public void doAccept() throws IOException {
        SocketChannel sc = serverSocketChannel.accept();

        if ( sc == null ) {
            logger.warning("bad hint");
            return;
        }

        sc.configureBlocking(false);
        var newKey = sc.register(selector, SelectionKey.OP_READ);
        var context = new Context(sc, this, newKey);
        newKey.attach(context);
    }

    private void silentlyClose(SelectionKey key) {
        Channel sc = (Channel) key.channel();
        try {
            sc.close();
        } catch (IOException e) {
            // ignore exception
        }
    }

    private void treatKey(SelectionKey key) {
        try {
            if (key.isValid() && key.isAcceptable()) {
                doAccept();
            }
        } catch(IOException ioe) {
            // lambda call in select requires to tunnel IOException
            throw new UncheckedIOException(ioe);
        }
        try {
            if (key.isValid() && key.isWritable()) {
                ((Context) key.attachment()).doWrite();
            }
            if (key.isValid() && key.isReadable()) {
                ((Context) key.attachment()).doRead();
            }
        } catch (IOException e) {
            logger.log(Level.INFO,"Connection closed with client due to IOException",e);
            silentlyClose(key);
        }
    }

    public void launch() throws IOException {
        serverSocketChannel.configureBlocking(false);
        serverSocketChannel.register(selector, SelectionKey.OP_ACCEPT);
        while(!Thread.interrupted()) {
            System.out.println("Starting select");
            try {
                selector.select(this::treatKey);
            } catch (UncheckedIOException tunneled) {
                throw tunneled.getCause();
            }
            System.out.println("Select finished");
        }
    }


    public static void main(String[] args) throws NumberFormatException, IOException {
        if (args.length!=1){
            usage();
            return;
        }
        new ServerChaton(Integer.parseInt(args[0])).launch();
    }

    private static void usage(){
        System.out.println("Usage : ServerChaton port");
    }
}
