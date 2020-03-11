package fr.upem.net.udp;


import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.AsynchronousCloseException;
import java.nio.channels.ClosedByInterruptException;
import java.nio.channels.DatagramChannel;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.*;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

public class ClientUpperCaseUDPFile {

        private static final Charset UTF8 = StandardCharsets.UTF_8;
        private static final int BUFFER_SIZE = 1024;
        private static final Logger logger = Logger.getLogger(ClientUpperCaseUDPRetry.class.getName());
        private static final ArrayBlockingQueue<String> blockingQueue = new ArrayBlockingQueue<>(10);
        private static void usage() {
            System.out.println("Usage : ClientUpperCaseUDPFile in-filename out-filename timeout host port ");
        }

        private static Thread makeAndRunFileReceiver(DatagramChannel datagramChannel) {
            var thread = new Thread(() -> {
                ByteBuffer byteBuffer = ByteBuffer.allocateDirect(BUFFER_SIZE);

                while (!Thread.interrupted()) {
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

                    try {
                        blockingQueue.put(UTF8.decode(byteBuffer).toString());
                    } catch (InterruptedException e) {
                        logger.log(Level.INFO, "Receiver interrupted.");
                        break;
                    }

                    byteBuffer.clear();
                }
            });

            thread.start();
            return thread;
        }

        private static void runFileSender(SocketAddress destination, List<String> lines, ArrayList<String> upperCaseLines,
                                          int timeout) throws IOException, InterruptedException {
            var datagramChannel = DatagramChannel.open();
            datagramChannel.bind(null);

            var thread = makeAndRunFileReceiver(datagramChannel);

            for( String line : lines ) {
                String result;
                do {
                    datagramChannel.send(UTF8.encode(line), destination);
                    result = blockingQueue.poll(timeout, TimeUnit.MILLISECONDS);
                } while (result == null);
                System.out.println("Received : "  + result );
                upperCaseLines.add(result);
            }

            thread.interrupt();
            datagramChannel.close();
        }


        public static void main(String[] args) throws IOException, InterruptedException {
            if (args.length !=5) {
                usage();
                return;
            }

            String inFilename = args[0];
            String outFilename = args[1];
            int timeout = Integer.valueOf(args[2]);
            String host=args[3];
            int port = Integer.valueOf(args[4]);
            SocketAddress dest = new InetSocketAddress(host,port);

            //Read all lines of inFilename opened in UTF-8
            List<String> lines= Files.readAllLines(Paths.get(inFilename),UTF8);
            ArrayList<String> upperCaseLines = new ArrayList<>();

			runFileSender(dest, lines, upperCaseLines, timeout);

            // Write upperCaseLines to outFilename in UTF-8
            Files.write(Paths.get(outFilename),upperCaseLines, UTF8,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.WRITE,
                    StandardOpenOption.TRUNCATE_EXISTING);
        }
    }


