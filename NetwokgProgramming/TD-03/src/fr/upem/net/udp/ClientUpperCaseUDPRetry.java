package fr.upem.net.udp;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.AsynchronousCloseException;
import java.nio.channels.DatagramChannel;
import java.nio.charset.Charset;
import java.util.Scanner;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ClientUpperCaseUDPRetry {

    private static final int BUFFER_SIZE = 1024;

    private static final Logger logger = Logger.getLogger(ClientUpperCaseUDPRetry.class.getName());

    private static final ArrayBlockingQueue<String> blockingQueue = new ArrayBlockingQueue<>(10);

    private static void usage(){
        System.out.println("Usage : NetcatUDP host port charset");
    }

    private static Thread makeAndRunReceiver(Charset charset, DatagramChannel datagramChannel) {
        var thread = new Thread(() -> {
            /* We can do the direct, it's alive all the live of the thread. */
            var byteBuffer = ByteBuffer.allocateDirect(1024);
            while(!Thread.interrupted()) {
                try {
                    datagramChannel.receive(byteBuffer);
                } catch (AsynchronousCloseException e) {
                    logger.log(Level.INFO, "Receiver interrupted.");
                    break;
                } catch (IOException e) {
                    logger.log(Level.SEVERE, "IOException", e);
                    break;
                }

                byteBuffer.flip();
                System.out.println("Received " + byteBuffer.remaining() + " bytes");
                var result = charset.decode(byteBuffer).toString();
                byteBuffer.clear();
                try {
                    blockingQueue.put(result);
                } catch (InterruptedException e) {
                    logger.log(Level.INFO, "Receiver interrupted.");
                    break;
                }
            }
        });

        thread.start();
        return thread;
    }

    private static void runConsoleSender(Charset charset, InetSocketAddress server) throws IOException, InterruptedException {
        try (Scanner scan = new Scanner(System.in)){
            ByteBuffer byteBuffer = ByteBuffer.allocate(BUFFER_SIZE);
            var datagramChannel = DatagramChannel.open();
            datagramChannel.bind(null);

            Thread receiver = makeAndRunReceiver(charset, datagramChannel);

            while(scan.hasNextLine()){
                String line = scan.nextLine();
                String result;
                do {
                    datagramChannel.send(charset.encode(line), server);
                    byteBuffer.clear();
                    result = blockingQueue.poll(1, TimeUnit.SECONDS);
                } while ( result == null );

                System.out.println("Received : "  + result );
            }

            receiver.interrupt();
            datagramChannel.close();
        }
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        if (args.length!=3){
            usage();
            return;
        }

        InetSocketAddress server = new InetSocketAddress(args[0],Integer.parseInt(args[1]));
        Charset cs = Charset.forName(args[2]);

        runConsoleSender(cs, server);
    }
}
