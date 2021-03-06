package fr.upem.net.tcp;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

public class OnDemandConcurrentLongSumServer {

    private static final Logger logger = Logger.getLogger(OnDemandConcurrentLongSumServer.class.getName());
    private static final int BUFFER_SIZE = 1024;
    private final ServerSocketChannel serverSocketChannel;

    public OnDemandConcurrentLongSumServer(int port) throws IOException {
        serverSocketChannel = ServerSocketChannel.open();
        serverSocketChannel.bind(new InetSocketAddress(port));
        logger.info(this.getClass().getName()
                + " starts on port " + port);
    }

    /**
     * Iterative server main loop
     *
     * @throws IOException
     */

    public void launch() throws IOException {
        logger.info("Server started");
            while(!Thread.interrupted()) {
                SocketChannel client = serverSocketChannel.accept();

                new Thread(() -> {
                    try {
                        logger.info("Connection accepted from " + client.getRemoteAddress());
                        serve(client);
                    } catch (IOException ioe) {
                        logger.log(Level.INFO,"Connection terminated with client by IOException",ioe.getCause());
                    } catch (InterruptedException ie) {
                        logger.info("Server interrupted");
                    } finally {
                        silentlyClose(client);
                    }
                }).start();
            }
    }

    /**
     * Treat the connection sc applying the protocole
     * All IOException are thrown
     *
     * @param sc
     * @throws IOException
     * @throws InterruptedException
     */
    private void serve(SocketChannel sc) throws IOException, InterruptedException{
        var STOPPED_INFO = "The server has stopped to serve the client.";
        ByteBuffer byteBuffer = ByteBuffer.allocate(BUFFER_SIZE);

        while ( true ) {
            /* Prepare the buffer to receive an int : */
            byteBuffer.limit(Integer.BYTES);

            /* Receive the int : */
            if ( !readFully(sc, byteBuffer) ){
                logger.info(STOPPED_INFO);
                return;
            }

            /* Pass the buffer in read mode : */
            byteBuffer.flip();

            /* Get the int : */
            int sumSize = byteBuffer.getInt();
            int sumByteSize = sumSize * Long.BYTES;

            /* Create the array that will contains all the Long : */
            var longs = new ArrayList<Long>(sumSize);

            /* Compact the buffer to optimize it and repass in write mode : */
            byteBuffer.compact();

            while ( sumSize > 0 ) {
                if ( sumByteSize < BUFFER_SIZE ) {
                    byteBuffer.limit(sumByteSize);
                }

                /* Receive the list of longs : */
                if ( !readFully(sc, byteBuffer) ){
                    logger.info(STOPPED_INFO);
                    break;
                }

                /* Get the longs in the buffer ans put them in the array : */
                byteBuffer.flip();
                while ( byteBuffer.hasRemaining()) {
                    longs.add(byteBuffer.getLong());
                    sumSize--;
                    sumByteSize -= Long.BYTES;
                }

                byteBuffer.clear();
            }

            /* Do the sum : */
            long sum = longs.stream().reduce(0L, Long::sum);
            byteBuffer.putLong(sum);

            /* Send the calculated sum to the client : */
            byteBuffer.flip();
            sc.write(byteBuffer);

            if ( byteBuffer.hasRemaining() ) {
                logger.info("Output stream closed");
                logger.info(STOPPED_INFO);
                break;
            }

            byteBuffer.clear();
        }

        logger.info("Serve succesfuly terminated");
	 }

    /**
     * Close a SocketChannel while ignoring IOExecption
     *
     * @param sc
     */

    private void silentlyClose(SocketChannel sc) {
        if (sc != null) {
            try {
                sc.close();
            } catch (IOException e) {
                // Do nothing
            }
        }
    }

    static boolean readFully(SocketChannel sc, ByteBuffer bb) throws IOException {
        while(bb.hasRemaining()) {
            if (sc.read(bb)==-1){
                logger.info("Input stream closed");
                return false;
            }
        }
        return true;
    }

    public static void main(String[] args) throws NumberFormatException, IOException {
        OnDemandConcurrentLongSumServer server = new OnDemandConcurrentLongSumServer(Integer.parseInt(args[0]));
        server.launch();
    }
}
