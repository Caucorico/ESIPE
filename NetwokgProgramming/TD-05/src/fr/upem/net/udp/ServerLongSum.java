package fr.upem.net.udp;

import java.io.IOException;
import java.net.BindException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.AlreadyBoundException;
import java.nio.channels.DatagramChannel;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ServerLongSum {
    public static final byte REQUEST_NUMBER = 1;
    public static final byte RESPONSE_ACKNOWLEDGEMENT = 2;
    public static final byte RESPONSE_SUM = 3;

    private static final Logger logger = Logger.getLogger(ServerLongSum.class.getName());
    private static final int BYTE_BUFFER_CAPACITY = 1024;
    private final DatagramChannel datagramChannel;
    private final ByteBuffer byteBuffer;
    private final HashMap<InetSocketAddress, HashMap<Long, SumData>>  data = new HashMap<>();


    public ServerLongSum(int port) throws IOException {
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
        System.out.println("Usage : ServerLongSum port");
    }

    public SumData getSumData(InetSocketAddress client, long sessionId, long totalOperandNumber) {
        var clientSession = data.computeIfAbsent(client, k -> new HashMap<>());
        return clientSession.computeIfAbsent(sessionId, k -> new SumData((int)totalOperandNumber));
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

    public boolean treatRequest(InetSocketAddress client) throws IOException {
        byteBuffer.flip();

        if ( byteBuffer.remaining() < Byte.BYTES ) {
            logger.log(Level.INFO, "Request ignored, wrong structure.");
            return false;
        }

        byte type = byteBuffer.get();
        if ( type != REQUEST_NUMBER ) {
            logger.log(Level.INFO, "Request ignored, wrong structure.");
            return false;
        }

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
            
            byteBuffer.flip();
            datagramChannel.send(byteBuffer, client);
        }

        buildAcknowledgement(sessionId, positionOperand);

        return true;
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

        if (args.length != 1) {
            usage();
            return;
        }

        ServerLongSum serverLongSum;
        int port = Integer.valueOf(args[0]);
        if (!(port >= 1024) & port <= 65535) {
            logger.severe("The port number must be between 1024 and 65535");
            return;
        }
        try {
            serverLongSum = new ServerLongSum(port);
        } catch (BindException e) {
            logger.severe("Server could not bind on " + port + "\nAnother server is probably running on this port.");
            return;
        }
        serverLongSum.serve();
    }

}
