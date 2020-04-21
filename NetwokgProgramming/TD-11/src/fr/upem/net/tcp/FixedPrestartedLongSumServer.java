package fr.upem.net.tcp;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.logging.Level;
import java.util.logging.Logger;

public class FixedPrestartedLongSumServer {

    private static final Logger logger = Logger.getLogger(OnDemandConcurrentLongSumServer.class.getName());
    private static final int BUFFER_SIZE = 1024;
    private final ServerSocketChannel serverSocketChannel;

    /**
     * The FixedPrestartedLongSumServer constructor.
     * It is in the constructor that we prepare the socket channel.
     *
     * @param port The port attributed to the server.
     *
     * @throws IOException The IOException can happen here if :
     *     - The port is already used.
     *     - The open doesn't work for any reason.
     */
    public FixedPrestartedLongSumServer(int port) throws IOException {
        serverSocketChannel = ServerSocketChannel.open();
        serverSocketChannel.bind(new InetSocketAddress(port));
        logger.info(this.getClass().getName()
                + " starts on port " + port);
    }

    private static boolean readFully(SocketChannel sc, ByteBuffer bb) {
        try {
            while(bb.hasRemaining()) {
                if (sc.read(bb)==-1){
                    logger.info("Input stream closed");
                    return false;
                }
            }
            return true;
        } catch(IOException e) {
            logger.info("ReadFully failed");
            return false;
        }
    }

    private static Optional<ArrayList<Long>> recoverLongSumProtocolNumbers(SocketChannel clientConnectedSocketChannel, ByteBuffer byteBuffer) {
        /* We don't trust the ByteBuffer content. So, we clear it : */
        byteBuffer.clear();

        /* Prepare the buffer to receive an int : */
        byteBuffer.limit(Integer.BYTES);

        /* We read the int with a ByteBuffer */
        if ( !readFully(clientConnectedSocketChannel, byteBuffer) ){
            return Optional.empty();
        }

        /* Pass the buffer in read mode : */
        byteBuffer.flip();

        /* Get the int : */
        int sumSize = byteBuffer.getInt();
        int sumByteSize = sumSize * Long.BYTES;

        /* Create the array that will contains all the Long : */
        var longs = new ArrayList<Long>(sumSize);

        /* Compact the buffer to optimize it and repass in write mode : */
        /* TODO : Ask to the teacher if a the compact is needed, and if it is better to send the limit to readFully instead modify it */
        byteBuffer.compact();

        while ( sumSize > 0 ) {
            if ( sumByteSize < BUFFER_SIZE ) {
                byteBuffer.limit(sumByteSize);
            }

            /* Receive the list of longs : */
            if ( !readFully(clientConnectedSocketChannel, byteBuffer) ){
                return Optional.empty();
            }

            /* Get the longs in the buffer and put them in the array : */
            byteBuffer.flip();
            while ( byteBuffer.hasRemaining()) {
                longs.add(byteBuffer.getLong());
                sumSize--;
                sumByteSize -= Long.BYTES;
            }

            byteBuffer.clear();
        }

        return Optional.of(longs);
    }

    private static boolean sendResultOfLongSumProtocol(SocketChannel clientConnectedSocketChannel, ByteBuffer byteBuffer, long sum) {
        /* We don't trust the ByteBuffer content. So, we clear it : */
        byteBuffer.clear();

        byteBuffer.putLong(sum);

        /* Send the calculated sum to the client : */
        byteBuffer.flip();

        try {
            clientConnectedSocketChannel.write(byteBuffer);
        } catch (IOException e) {
            logger.info("The write to the client failed.");
            return false;
        }

        if ( byteBuffer.hasRemaining() ) {
            logger.info("The write to the client failed.");
            return false;
        }

        return true;
    }

    /**
     * Do the long sum server protocol and return the sum.
     * In first, we receive all the number to sum.
     * In second, we do the sum.
     * Finally, we send the sum.
     */
    private static boolean doLongSumServerProtocol(SocketChannel clientConnectedSocketChannel, ByteBuffer byteBuffer) {
        /* We read all the numbers send by the client :*/
        var optionalNumbers = recoverLongSumProtocolNumbers(clientConnectedSocketChannel, byteBuffer);

        if ( optionalNumbers.isEmpty() ) {
            return false;
        }
        var numbers = optionalNumbers.get();

        /* We compute the sum of the numbers : */
        long sum = numbers.stream().reduce(0L, Long::sum);

        /* When the sum is computed, we can send it to the client : */
        return sendResultOfLongSumProtocol(clientConnectedSocketChannel, byteBuffer, sum);
    }

    /**
     * This function return the Runnable run by the server threads.
     * This Runnable do the accept !
     *
     * @param serverSocketChannel The server socket channel. ( Get with the ServerSocketChannel.open() ).
     *
     * @return The Runnable.
     */
    private static Runnable getTreatRunnable(ServerSocketChannel serverSocketChannel) {
        return () -> {
            /* We allocate a single buffer for each threads : */
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(BUFFER_SIZE);

            /*  Main loop :  */
            while ( !Thread.interrupted() ) {

                SocketChannel clientConnectedSocketChannel = null;
                try {
                    /* First, we need to wait a request with the accept : */
                    clientConnectedSocketChannel = serverSocketChannel.accept();
                } catch (IOException e) {
                    /* TODO : Ask to the teacher how to catch these exceptions :
                    *    - ClosedChannelException. ( Close the thread if the channel is close ? )
                    *    - AsynchronousCloseException ( Do I try to close the SocketChannel in this situation ? )
                    *    - ClosedByInterruptException ( Do I try to close the SocketChannel in this situation ? )
                    */
                    logger.severe("The accept failed, the thread is broken, we stopp it");
                    return;
                }

                /* this is not the server that close the connection, so we can only wait : */
                while ( true ) {
                    /* On each client request, we need to do the protocol : */
                    var res = doLongSumServerProtocol(clientConnectedSocketChannel, byteBuffer);

                    /* If the protocol return false, wa can stop the connection with the client. */
                    if ( !res ) break;
                }

                /* We close the connection */
                try {
                    clientConnectedSocketChannel.close();
                } catch (IOException e) {
                    /* TODO : Ask to the teacher how to catch these exceptions :
                     *    - IOException. ( If the close fail, should we close the thread or try to continue ? )
                     */
                    logger.severe("The close failed, the thread is broken, we stopp it");
                    return;
                }
            }
        };
    }

    /**
     * This function run all the thread needed by the server.
     * The "threadNumber" threads are run directly and wait a request.
     *
     * @param threadNumber The max number of thread.
     */
    public void launch(int threadNumber) throws InterruptedException {
        logger.info("Server started");

        ArrayList<Thread> threads = new ArrayList<>();
        for ( var i = 0 ; i < threadNumber ; i++ ) {
            Thread t = new Thread(getTreatRunnable(serverSocketChannel));
            threads.add(t);
            t.start();
        }

        /* Wait the threads : */
        for ( var thread : threads) {
            thread.join();
        }
    }

    public static void main(String[] args) throws NumberFormatException, IOException, InterruptedException {
        if ( args.length != 2 ) {
            System.err.println("Usage : java class_name <port> <max_thread>");
            return;
        }
        FixedPrestartedLongSumServer server = new FixedPrestartedLongSumServer(Integer.parseInt(args[0]));
        server.launch(Integer.parseInt(args[1]));
    }
}
