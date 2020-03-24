package fr.upem.net.udp;

import java.nio.charset.StandardCharsets;
import java.util.logging.Logger;
import java.io.IOException;
import java.net.BindException;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.DatagramChannel;
import java.nio.charset.Charset;

public class ServerIdUpperCaseUDP {

    private static final Logger logger = Logger.getLogger(ServerIdUpperCaseUDP.class.getName());
    private static final int BUFFER_SIZE = 1024;
    private final static Charset UTF8 = StandardCharsets.UTF_8;
    private final DatagramChannel dc;
    private final ByteBuffer buff = ByteBuffer.allocateDirect(BUFFER_SIZE);

    public ServerIdUpperCaseUDP(int port) throws IOException {
        dc = DatagramChannel.open();
        dc.bind(new InetSocketAddress(port));
        logger.info("ServerBetterUpperCaseUDP started on port " + port);
    }

    public void serve() throws IOException {
        while (!Thread.interrupted()) {
            buff.clear();
            InetSocketAddress sender = (InetSocketAddress) dc.receive(buff);
            buff.flip();

            /* If the request is to small, ignore it. */
            if (buff.remaining() < Long.BYTES ) {
                continue;
            }

            /* Get the request infos : */
            var id = buff.getLong();
            var message = UTF8.decode(buff).toString();

            /* Put message in uppercase */
            message = message.toUpperCase();
            buff.clear();
            buff.putLong(id);
            buff.put(UTF8.encode(message));
            buff.flip();
            dc.send(buff, sender);
        }
    }

    public static void usage() {
        System.out.println("Usage : ServerIdUpperCaseUDP port");
    }

    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            usage();
            return;
        }
        ServerIdUpperCaseUDP server;
        int port = Integer.valueOf(args[0]);
        if (!(port >= 1024) & port <= 65535) {
            logger.severe("The port number must be between 1024 and 65535");
            return;
        }
        try {
            server = new ServerIdUpperCaseUDP(port);
        } catch (BindException e) {
            logger.severe("Server could not bind on " + port + "\nAnother server is probably running on this port.");
            return;
        }
        server.serve();
    }
}
