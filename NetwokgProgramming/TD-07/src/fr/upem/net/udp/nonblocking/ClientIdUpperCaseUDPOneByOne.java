package fr.upem.net.udp.nonblocking;


import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.DatagramChannel;
import java.nio.channels.SelectionKey;
import java.nio.channels.Selector;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.time.Instant;
import java.time.temporal.TemporalUnit;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

public class ClientIdUpperCaseUDPOneByOne {

    private static Logger logger = Logger.getLogger(ClientIdUpperCaseUDPOneByOne.class.getName());
    private static final Charset UTF8 = Charset.forName("UTF8");
    private static final int BUFFER_SIZE = 1024;

    private enum State {SENDING, RECEIVING, FINISHED};

    private final List<String> lines;
    private final List<String> upperCaseLines = new ArrayList<>();
    private final int timeout;
    private final InetSocketAddress serverAddress;
    private final DatagramChannel dc;
    private final Selector selector;
    private final SelectionKey uniqueKey;

    private final ByteBuffer byteBuffer;
    private Instant lastSend;
    private long currentId;

    private State state;

    private static void usage() {
        System.out.println("Usage : ClientIdUpperCaseUDPOneByOne in-filename out-filename timeout host port ");
    }

    public ClientIdUpperCaseUDPOneByOne(List<String> lines, int timeout, InetSocketAddress serverAddress) throws IOException {
        this.lines = lines;
        this.timeout = timeout;
        this.serverAddress = serverAddress;
        this.dc = DatagramChannel.open();
        dc.configureBlocking(false);
        dc.bind(null);
        this.selector = Selector.open();
        this.uniqueKey = dc.register(selector, SelectionKey.OP_WRITE);
        this.state = State.SENDING;
        this.byteBuffer = ByteBuffer.allocateDirect(BUFFER_SIZE);
        this.lastSend = Instant.now().minusMillis(timeout);
        this.currentId = 0;
    }


    public static void main(String[] args) throws IOException, InterruptedException {
        if (args.length != 5) {
            usage();
            return;
        }

        String inFilename = args[0];
        String outFilename = args[1];
        int timeout = Integer.valueOf(args[2]);
        String host = args[3];
        int port = Integer.valueOf(args[4]);
        InetSocketAddress serverAddress = new InetSocketAddress(host, port);

        //Read all lines of inFilename opened in UTF-8
        List<String> lines = Files.readAllLines(Paths.get(inFilename), UTF8);
        //Create client with the parameters and launch it
        ClientIdUpperCaseUDPOneByOne client = new ClientIdUpperCaseUDPOneByOne(lines, timeout, serverAddress);
        List<String> upperCaseLines = client.launch();
        Files.write(Paths.get(outFilename), upperCaseLines, UTF8,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING);

    }

    
    private List<String> launch() throws IOException, InterruptedException {
        Set<SelectionKey> selectedKeys = selector.selectedKeys();
        while (!isFinished()) {
            logger.info("Waiting");
            selector.select(updateInterestOps());
            for (SelectionKey key : selectedKeys) {
                if (key.isValid() && key.isWritable()) {
                    doWrite();
                }
                if (key.isValid() && key.isReadable()) {
                    doRead();
                }
            }

            selectedKeys.clear();
        }
        dc.close();
        return upperCaseLines;
    }

    /**
    * Updates the interestOps on key based on state of the context
    *
    * @return the timeout for the next select (0 means no timeout)
    */
    
    private int updateInterestOps() {
        long timeout = 0;
        switch (state) {
            case RECEIVING:
                uniqueKey.interestOps(SelectionKey.OP_READ);
                timeout = Duration.between(Instant.now(), lastSend.plusMillis(this.timeout)).toMillis();
                if ( timeout <= 0 ) {
                    state = State.SENDING;
                    uniqueKey.interestOps(SelectionKey.OP_WRITE);
                    return 0;
                }
                return (int)timeout;
            case SENDING:
                uniqueKey.interestOps(SelectionKey.OP_WRITE);
                return 0;
            default:
                return this.timeout;
        }
    }

    private boolean isFinished() {
        return state == State.FINISHED;
    }

    /**
    * Performs the receptions of packets
    *
    * @throws IOException
    */

    private void doRead() throws IOException {
        byteBuffer.clear();
        logger.info("Try receive");
        var exp = dc.receive(byteBuffer);

        if ( exp == null ) return;
        byteBuffer.flip();
        if ( byteBuffer.remaining() < Long.BYTES ) return;

        long id = byteBuffer.getLong();

        /* TODO : NOT RESEND IN THIS CASE */
        if ( id != currentId ) return;

        upperCaseLines.add(UTF8.decode(byteBuffer).toString());
        currentId++;

        if ( upperCaseLines.size() == lines.size() ) {
            state = State.FINISHED;
            return;
        }

        state = State.SENDING;
    }

    /**
    * Tries to send the packets
    *
    * @throws IOException
    */

    private void doWrite() throws IOException {
        byteBuffer.clear();

        byteBuffer.putLong(currentId);
        byteBuffer.put(UTF8.encode(lines.get((int)currentId)));
        byteBuffer.flip();

        logger.info("Try send");
        dc.send(byteBuffer, serverAddress);
        if ( byteBuffer.hasRemaining() ) return;
        lastSend = Instant.now();

        state = State.RECEIVING;
    }
}







