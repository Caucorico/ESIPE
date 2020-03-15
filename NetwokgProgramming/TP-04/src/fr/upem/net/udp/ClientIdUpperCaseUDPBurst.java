package fr.upem.net.udp;


import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.AsynchronousCloseException;
import java.nio.channels.DatagramChannel;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ClientIdUpperCaseUDPBurst {

        private static Logger logger = Logger.getLogger(ClientIdUpperCaseUDPBurst.class.getName());
        private static final Charset UTF8 = StandardCharsets.UTF_8;
        private static final int BUFFER_SIZE = 1024;
        private final List<String> lines;
        private final int nbLines;
        private final String[] upperCaseLines; //
        private final int timeout;
        private final String outFilename;
        private final InetSocketAddress serverAddress;
        private final DatagramChannel dc;
        private final BitSet received;         // BitSet marking received requests
        private final BitSet mask;

        private static final Object lock = new Object();

        private static void usage() {
            System.out.println("Usage : ClientIdUpperCaseUDPBurst in-filename out-filename timeout host port ");
        }

        private ClientIdUpperCaseUDPBurst(List<String> lines,int timeout,InetSocketAddress serverAddress,String outFilename) throws IOException {
            this.lines = lines;
            this.nbLines = lines.size();
            this.timeout = timeout;
            this.outFilename = outFilename;
            this.serverAddress = serverAddress;
            this.dc = DatagramChannel.open();
            dc.bind(null);
            this.received= new BitSet(nbLines);
            this.mask = new BitSet(nbLines);
            for ( var i = 0 ; i < nbLines ; i++ ) {
                this.mask.set(i);
            }
            this.upperCaseLines = new String[nbLines];
        }

        private void senderThreadRun() {
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(BUFFER_SIZE);

            try {
                while (!Thread.interrupted()) {
                    synchronized (lock) {
                        for (int i = 0; i < lines.size(); i++) {
                            if ( !received.get(i)) {
                                byteBuffer.putLong(i);
                                byteBuffer.put(UTF8.encode(lines.get(i)));
                                byteBuffer.flip();
                                dc.send(byteBuffer, serverAddress);
                                byteBuffer.clear();
                            }
                        }
                    }
                    Thread.sleep(timeout);
                }
            } catch (AsynchronousCloseException|InterruptedException e) {
                logger.log(Level.INFO, "Sender stopped.");
            } catch (IOException e) {
                logger.log(Level.SEVERE, "Unexpected IOException.");
            }
        }

        private void launch() throws IOException {
            Thread senderThread = new Thread(this::senderThreadRun);
            senderThread.start();
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(BUFFER_SIZE);
            
            while ( true ) {
                dc.receive(byteBuffer);
                byteBuffer.flip();

                synchronized (lock) {
                    int i = (int)byteBuffer.getLong();
                    upperCaseLines[i] = UTF8.decode(byteBuffer).toString();
                    received.set(i);
                    if ( received.equals(mask) ) {
                        break;
                    }
                }

                byteBuffer.clear();
            }

            senderThread.interrupt();
            dc.close();

            Files.write(Paths.get(outFilename),Arrays.asList(upperCaseLines), UTF8,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING);

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
            InetSocketAddress serverAddress = new InetSocketAddress(host,port);

            //Read all lines of inFilename opened in UTF-8
            List<String> lines= Files.readAllLines(Paths.get(inFilename),UTF8);
            //Create client with the parameters and launch it
            ClientIdUpperCaseUDPBurst client = new ClientIdUpperCaseUDPBurst(lines,timeout,serverAddress,outFilename);
            client.launch();

        }
    }


