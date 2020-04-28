package fr.upem.net.tcp;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.AsynchronousCloseException;
import java.nio.channels.ServerSocketChannel;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

public class FixedPrestartedLongSumServerWithTimeout {

    private static final Logger logger = Logger.getLogger(OnDemandConcurrentLongSumServer.class.getName());
    private static final int BUFFER_SIZE = 1024;
    private final ServerSocketChannel serverSocketChannel;
    private final ConcurrentHashMap<Thread, ThreadData> threads = new ConcurrentHashMap<>();
    private final int timeout;

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
    public FixedPrestartedLongSumServerWithTimeout(int port, int timeout) throws IOException {
        serverSocketChannel = ServerSocketChannel.open();
        serverSocketChannel.bind(new InetSocketAddress(port));
        logger.info(this.getClass().getName()
                + " starts on port " + port);
        this.timeout = timeout;
    }

    private boolean readFully(ByteBuffer bb) {
        try {
            while(bb.hasRemaining()) {
                if (threads.get(Thread.currentThread()).getSocketChannel().read(bb)==-1){
                    logger.info("Input stream closed");
                    return false;
                }

                threads.get(Thread.currentThread()).tick();
            }
            return true;
        } catch(IOException e) {
            logger.info("ReadFully failed");
            return false;
        }
    }

    private Optional<Long> recoverLongSumProtocolNumbers(ByteBuffer byteBuffer) {
        /* We don't trust the ByteBuffer content. So, we clear it : */
        byteBuffer.clear();

        /* Prepare the buffer to receive an int : */
        byteBuffer.limit(Integer.BYTES);

        /* We read the int with a ByteBuffer */
        if ( !readFully(byteBuffer) ){
            return Optional.empty();
        }

        /* Pass the buffer in read mode : */
        byteBuffer.flip();

        /* Get the int : */
        int sumSize = byteBuffer.getInt();
        int sumByteSize = sumSize * Long.BYTES;

        /* Create the sum : */
        var sum = 0L;

        /* Compact the buffer to optimize it and repass in write mode : */
        /* TODO : Ask to the teacher if a the compact is needed */
        byteBuffer.compact();

        while ( sumSize > 0 ) {
            if ( sumByteSize < BUFFER_SIZE ) {
                byteBuffer.limit(sumByteSize);
            }

            /* Receive the list of longs : */
            if ( !readFully(byteBuffer) ){
                return Optional.empty();
            }

            /* Get the longs in the buffer and put them in the array : */
            byteBuffer.flip();
            while ( byteBuffer.hasRemaining()) {
                sum += byteBuffer.getLong();
                sumSize--;
                sumByteSize -= Long.BYTES;
            }

            byteBuffer.clear();
        }

        return Optional.of(sum);
    }

    private boolean sendResultOfLongSumProtocol(ByteBuffer byteBuffer, long sum) {
        /* We don't trust the ByteBuffer content. So, we clear it : */
        byteBuffer.clear();

        byteBuffer.putLong(sum);

        /* Send the calculated sum to the client : */
        byteBuffer.flip();

        try {
            threads.get(Thread.currentThread()).getSocketChannel().write(byteBuffer);
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
    private void serve(ByteBuffer byteBuffer) {

        /* this is not the server that close the connection, so we can only wait : */
        while( true ) {
            /* We read all the numbers send by the client :*/
            var optionalNumber = recoverLongSumProtocolNumbers(byteBuffer);

            if ( optionalNumber.isEmpty() ) {
                break;
            }
            var sum = optionalNumber.get();

            /* When the sum is computed, we can send it to the client : */
            var res = sendResultOfLongSumProtocol(byteBuffer, sum);
            if (!res) {
                break;
            }
        }

        logger.info("WorkerThread stop protocol.");
    }

    public Thread runTimeoutThread() {
        Thread t = new Thread(() -> {
            while ( !Thread.interrupted() ) {
                try {
                    for (var entry : threads.entrySet() ) {
                        entry.getValue().closeIfInactive(timeout);
                    }
                } catch (IOException e) {
                    logger.log(Level.SEVERE, "TimeoutThread failed to close a connection", e.getCause());
                    return;
                }

                try {
                    Thread.sleep(timeout);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }

        });
        t.start();
        return t;
    }

    /**
     * This function run all the thread needed by the server.
     * The "threadNumber" threads are run directly and wait a request.
     *
     * @param threadNumber The max number of thread.
     */
    public void launch(int threadNumber) throws InterruptedException {
        logger.info("Server started");

        for ( var i = 0 ; i < threadNumber ; i++ ) {
            Thread t = new Thread(() -> {
                /* We allocate a single buffer for each threads : */
                ByteBuffer byteBuffer = ByteBuffer.allocateDirect(BUFFER_SIZE);

                try {
                    /*  Main loop :  */
                    while ( !Thread.interrupted() ) {

                        /* First, we need to wait a request with the accept : */
                        var clientConnectedSocketChannel = serverSocketChannel.accept();
                        threads.get(Thread.currentThread()).setSocketChannel(clientConnectedSocketChannel);
                        /* On each client request, we need to do the protocol : */
                        serve(byteBuffer); /* <=> serve */

                        /* TODO : Ask to the teacher if it's better to use condition/optional or Exception */
                    }
                } catch (AsynchronousCloseException e) {
                    logger.info("Thread closed by AsynchronousCloseException.");
                } catch (IOException e) {
                    logger.log(Level.SEVERE, "Thead closed by IOException.");
                }
            });
            threads.put(t, new ThreadData());
            t.start();
        }

        var timeoutThread = runTimeoutThread();

        /* Wait the threads : */
        for ( var thread : threads.keySet()) {
            thread.join();
        }

        timeoutThread.join();
    }

    public static void main(String[] args) throws NumberFormatException, IOException, InterruptedException {
        if ( args.length != 3 ) {
            System.err.println("Usage : java class_name <port> <max_thread> <timeout>");
            return;
        }
        FixedPrestartedLongSumServerWithTimeout server = new FixedPrestartedLongSumServerWithTimeout(Integer.parseInt(args[0]), Integer.parseInt(args[2]));
        server.launch(Integer.parseInt(args[1]));
    }
}
