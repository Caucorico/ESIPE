package fr.upem.net.udp;

import java.io.IOException;
import java.net.BindException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.AlreadyBoundException;
import java.nio.channels.DatagramChannel;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ServerFreeLongSum {
    public static final byte REQUEST_NUMBER = 1;
    public static final byte RESPONSE_ACKNOWLEDGEMENT = 2;
    public static final byte RESPONSE_SUM = 3;
    public static final byte REQUEST_CLEAR = 4;
    public static final byte RESPONSE_CLEAR = 5;

    private final int timeout;
    private static final Logger logger = Logger.getLogger(ServerFreeLongSum.class.getName());
    private static final int BYTE_BUFFER_CAPACITY = 1024;
    private final DatagramChannel datagramChannel;
    private final ByteBuffer byteBuffer;
    private final ConcurrentHashMap<InetSocketAddress, ConcurrentHashMap<Long, SumData>> data = new ConcurrentHashMap<>();


    public ServerFreeLongSum(int port, int timeout) throws IOException {
        this.timeout = timeout;
        datagramChannel = DatagramChannel.open();
        try {
            datagramChannel.bind(new InetSocketAddress(port));
        } catch (AlreadyBoundException e) {
            logger.log(Level.SEVERE, "The port " + port + "is already used.");
            throw e;
        }

        byteBuffer = ByteBuffer.allocateDirect(BYTE_BUFFER_CAPACITY);
        logger.info("ServerLongSum started on port " + port);
    }

    public static void usage() {
        System.out.println("Usage : ServerLongSum port timeout(ms)");
    }

    private Thread runThreadTimeout(InetSocketAddress client, long sessionId) {
        Thread thread = new Thread(() -> {
            try {
                Thread.sleep(timeout);
            } catch (InterruptedException e) {
                return;
            }
            removeSumData(client, sessionId);
            logger.log(Level.INFO, "Timedout");
        });
        thread.start();
        return thread;
    }

    public void removeSumData(InetSocketAddress client, long sessionId) {
//        data.computeIfPresent(client, (k, v) -> {
//            v.remove(sessionId);
//            return null;
//        });
        data.get(client).remove(sessionId);
    }

    public SumData getSumData(InetSocketAddress client, long sessionId, long totalOperandNumber) {
        var clientSession = data.computeIfAbsent(client, k -> new ConcurrentHashMap<>());
        return clientSession.computeIfAbsent(sessionId, k -> {
            runThreadTimeout(client, sessionId);
            return new SumData((int)totalOperandNumber);
        });
    }

    public void buildResponse(SumData sumData, long sessionId) {
        byteBuffer.clear();
        byteBuffer.put(RESPONSE_SUM);
        byteBuffer.putLong(sessionId);
        byteBuffer.putLong(sumData.getSum());
    }

    public void buildAcknowledgement(long sessionId, long positionOperand) {
        byteBuffer.clear();
        byteBuffer.put(RESPONSE_ACKNOWLEDGEMENT);
        byteBuffer.putLong(sessionId);
        byteBuffer.putLong(positionOperand);
    }

    public void buildClearAcknowledgement(long sessionId) {
        byteBuffer.clear();
        byteBuffer.put(RESPONSE_CLEAR);
        byteBuffer.putLong(sessionId);
    }

    public boolean treatPutRequest(InetSocketAddress client) throws IOException {
        if ( byteBuffer.remaining() < Long.BYTES ) {
            logger.log(Level.INFO, "Request ignored, wrong structure.");
            return false;
        }

        long sessionId = byteBuffer.getLong();

        if ( byteBuffer.remaining() < Long.BYTES ) {
            logger.log(Level.INFO, "Request ignored, wrong structure.");
            return false;
        }

        long positionOperand = byteBuffer.getLong();
        if ( positionOperand < 0 ) {
            logger.log(Level.INFO, "Request ignored, wrong structure.");
            return false;
        }

        if ( byteBuffer.remaining() < Long.BYTES ) {
            logger.log(Level.INFO, "Request ignored, wrong structure.");
            return false;
        }

        long totalOperandNumber = byteBuffer.getLong();
        if ( totalOperandNumber < 0 ) {
            logger.log(Level.INFO, "Request ignored, wrong structure.");
            return false;
        }

        if ( byteBuffer.remaining() < Long.BYTES ) {
            logger.log(Level.INFO, "Request ignored, wrong structure.");
            return false;
        }

        long operand = byteBuffer.getLong();

        var sumData = getSumData(client, sessionId, totalOperandNumber);
        sumData.addOperand((int)positionOperand, operand);

        if ( sumData.isFull() ) {
            buildResponse(sumData, sessionId);
        } else {
            buildAcknowledgement(sessionId, positionOperand);
        }

        return true;
    }

    public boolean treatClearRequest(InetSocketAddress client) {
        if ( byteBuffer.remaining() < Long.BYTES ) {
            logger.log(Level.INFO, "Request ignored, wrong structure.");
            return false;
        }

        var sessionId = byteBuffer.getLong();

        removeSumData(client, sessionId);
        buildClearAcknowledgement(sessionId);

        return true;
    }

    public boolean treatRequest(InetSocketAddress client) throws IOException {
        byteBuffer.flip();

        if ( byteBuffer.remaining() < Byte.BYTES ) {
            logger.log(Level.INFO, "Request ignored, wrong structure.");
            return false;
        }

        byte type = byteBuffer.get();

        switch ( type ) {
            case REQUEST_NUMBER:
                return treatPutRequest(client);
            case REQUEST_CLEAR:
                return treatClearRequest(client);
            default:
                logger.log(Level.INFO, "Request ignored, wrong structure.");
                return false;
        }

    }

    public void serve() throws IOException {
        while (!Thread.interrupted()) {
            byteBuffer.clear();
            var client = (InetSocketAddress)datagramChannel.receive(byteBuffer);
            if ( treatRequest(client) ) {
                byteBuffer.flip();
                datagramChannel.send(byteBuffer, client);
            }
        }
    }

    public static void main(String[] args) throws IOException {

        if (args.length != 2) {
            usage();
            return;
        }

        ServerFreeLongSum serverLongSum;
        int port = Integer.valueOf(args[0]);
        if (!(port >= 1024) & port <= 65535) {
            logger.severe("The port number must be between 1024 and 65535");
            return;
        }
        int timeout = Integer.valueOf(args[1]);
        if ( timeout < 0 ) {
            logger.severe("The timout must be positive !");
            return;
        }


        try {
            serverLongSum = new ServerFreeLongSum(port, timeout);
        } catch (BindException e) {
            logger.severe("Server could not bind on " + port + "\nAnother server is probably running on this port.");
            return;
        }
        serverLongSum.serve();
    }

}
