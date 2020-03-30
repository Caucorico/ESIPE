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
import java.util.*;
import java.util.logging.Logger;

public class ClientIdUpperCaseUDPBurst {

    private static Logger logger = Logger.getLogger(ClientIdUpperCaseUDPOneByOne.class.getName());
    private static final Charset UTF8 = Charset.forName("UTF8");
    private static final int BUFFER_SIZE = 1024;

    private enum State {SENDING, RECEIVING, FINISHED};

    private final List<String> lines;
    private final String[] upperCaseLines;
    private final int timeout;
    private final InetSocketAddress serverAddress;
    private final DatagramChannel dc;
    private final Selector selector;
    private final SelectionKey uniqueKey;
    private final ByteBuffer byteBuffer;
    private final BitSet bitSet;

    private int sentLotNumber;
    private Instant lastSend;
    private int lastPos;

    private State state;

    private static void usage() {
        System.out.println("Usage : ClientIdUpperCaseUDPOneByOne in-filename out-filename timeout host port ");
    }

    public ClientIdUpperCaseUDPBurst(List<String> lines, int timeout, InetSocketAddress serverAddress) throws IOException {
        this.lines = lines;
        this.timeout = timeout;
        this.serverAddress = serverAddress;
        this.dc = DatagramChannel.open();
        dc.configureBlocking(false);
        dc.bind(null);
        this.selector = Selector.open();
        this.uniqueKey = dc.register(selector, SelectionKey.OP_WRITE);
        this.state = State.SENDING;
        this.byteBuffer  = ByteBuffer.allocateDirect(BUFFER_SIZE);
        this.bitSet = new BitSet(lines.size());
        this.upperCaseLines = new String[lines.size()];

        this.sentLotNumber = 0;
        this.lastSend = Instant.now().minusMillis(timeout);
        this.lastPos = 0;
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
        ClientIdUpperCaseUDPBurst client = new ClientIdUpperCaseUDPBurst(lines, timeout, serverAddress);
        List<String> upperCaseLines = client.launch();
        Files.write(Paths.get(outFilename), upperCaseLines, UTF8,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING);

    }

    
    private List<String> launch() throws IOException, InterruptedException {
        Set<SelectionKey> selectedKeys = selector.selectedKeys();
        while (!isFinished()) {
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
        return Arrays.asList(upperCaseLines);
    }

    /**
    * Updates the interestOps on key based on state of the context
    *
    * @return the timeout for the next select (0 means no timeout)
    */
    
    private int updateInterestOps() {
        switch (state) {
            case RECEIVING:
                var delay = Duration.between(Instant.now(), lastSend.plusMillis(timeout)).toMillis();
                if ( delay <= 0 ) {
                    state = State.SENDING;
                    uniqueKey.interestOps(SelectionKey.OP_WRITE);
                    return 0;
                }
                uniqueKey.interestOps(SelectionKey.OP_READ);
                return (int)delay;
            case SENDING:
            default:
                uniqueKey.interestOps(SelectionKey.OP_WRITE);
                return 0;
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
        var exp = dc.receive(byteBuffer);
        if ( exp == null ) return;
        byteBuffer.flip();

        if ( byteBuffer.remaining() < Long.BYTES ) return;

        long id = byteBuffer.getLong();
        upperCaseLines[(int)id] = UTF8.decode(byteBuffer).toString();
        bitSet.set((int)id);

        if (  bitSet.cardinality() == lines.size() ) {
            state = State.FINISHED;
        }

    }

    /**
    * Tries to send the packets
    *
    * @throws IOException
    */

    private void doWrite() throws IOException {
        var i = bitSet.nextClearBit(lastPos);
        byteBuffer.clear();
        byteBuffer.putLong(i);
        byteBuffer.put(UTF8.encode(lines.get(i)));
        byteBuffer.flip();

        dc.send(byteBuffer, serverAddress);
        if ( !byteBuffer.hasRemaining() ) {
            sentLotNumber++;
            lastPos = i+1;

            if ( sentLotNumber == (lines.size()-bitSet.cardinality()) ) {
                sentLotNumber = 0;
                lastPos = 0;
                state = State.RECEIVING;
                lastSend = Instant.now();
                return;
            }
        }
    }
}







