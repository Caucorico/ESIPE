package fr.upem.net.tcp;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.SocketChannel;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.logging.Logger;

public class ClientConcatenation {

    private static final Logger logger = Logger.getLogger(ClientConcatenation.class.getName());
    private static final int BUFFER_SIZE = 1024;
    public static final Charset UTF8_CHARSET = StandardCharsets.UTF_8;

    private final ByteBuffer byteBuffer;
    private final InetSocketAddress server;


    public ClientConcatenation(String serverAddress, int port) {
        Objects.requireNonNull(serverAddress, "serverAdress cannot be null !");
        if ( serverAddress.isEmpty() ) {
            throw new IllegalArgumentException("The server address cannot be emty !");
        }

        if ( port <= 0 || port >= 65536 ) {
            throw new IllegalArgumentException("The server port needs to be a valid port !");
        }

        byteBuffer = ByteBuffer.allocateDirect(BUFFER_SIZE);
        server = new InetSocketAddress(serverAddress, port);
    }

    private static Optional<Integer> readInt(ByteBuffer byteBuffer, SocketChannel channel) throws IOException {
        byteBuffer.clear();
        byteBuffer.limit(Integer.BYTES);

        while ( byteBuffer.hasRemaining() ) {
            var res = channel.read(byteBuffer);
            if ( res == -1 ) {
                byteBuffer.limit(BUFFER_SIZE);
                return Optional.empty();
            }
        }

        byteBuffer.flip();

        /* reset the buffer size :  */
        /* TODO : in finally bloc ? */
        byteBuffer.limit(BUFFER_SIZE);
        return Optional.of(byteBuffer.getInt());
    }

    private static Optional<String> readLimitedString(ByteBuffer byteBuffer, SocketChannel channel, int responseByteSize) throws IOException {
        StringBuilder sb = new StringBuilder();
        byteBuffer.clear();

        while ( responseByteSize > 0 ) {
            var res = channel.read(byteBuffer);
            if ( res == -1 ) {
                return Optional.empty();
            }

            byteBuffer.flip();
            var responseLength = byteBuffer.remaining();
            sb.append(UTF8_CHARSET.decode(byteBuffer));
            responseByteSize -= responseLength;
            byteBuffer.clear();
        }

        return Optional.of(sb.toString());
    }

    private static Optional<String> receiveStringResponse(ByteBuffer byteBuffer, SocketChannel socketChannel) throws IOException {
        Optional<Integer> optionalSize = readInt(byteBuffer, socketChannel);
        if ( optionalSize.isEmpty() ) return Optional.empty();
        logger.info("String size received ! : " + optionalSize.get());
        return readLimitedString(byteBuffer, socketChannel, optionalSize.get());
    }

    public Optional<String> requestConcatFromList(List<String> strings) throws IOException {
        try (var channel = SocketChannel.open(server) ) {
            byteBuffer.putInt(strings.size());
            byteBuffer.flip();
            channel.write(byteBuffer);
            for (String sub : strings ) {
                byteBuffer.clear();
                var bb = UTF8_CHARSET.encode(sub);
                byteBuffer.putInt(bb.remaining());
                byteBuffer.flip();
                channel.write(byteBuffer);
                channel.write(bb);
                byteBuffer.clear();
            }
            channel.shutdownOutput();
            Optional<String> response = receiveStringResponse(byteBuffer, channel);
            channel.shutdownInput();
            return response;
        }
    }

    public static void usage() {
        System.err.println("Usage : ClientConcatenation <address> <port>");
    }


    public static void main(String[] args) throws IOException {
        if ( args.length != 2 ) {
            usage();
            return;
        }

        String server = args[0];
        int port = Integer.valueOf(args[1]);
        ClientConcatenation clientConcatenation = new ClientConcatenation(server, port);

        Scanner scan = new Scanner(System.in);
        ArrayList<String> strings = new ArrayList<>();

        while(scan.hasNextLine()){
            strings.add(scan.nextLine());
        }

        System.out.println(strings.size());

        System.out.println(clientConcatenation.requestConcatFromList(strings));
    }


}
