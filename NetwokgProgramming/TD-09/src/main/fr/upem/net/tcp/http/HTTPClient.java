package fr.upem.net.tcp.http;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.SocketChannel;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;

public class HTTPClient {

    private final static Charset ASCII = StandardCharsets.US_ASCII;
    private final static Charset ISO = StandardCharsets.ISO_8859_1;

    private final InetSocketAddress serverAddress;

    private final String resource;

    public HTTPClient(String address, String resource) throws IOException {
        this.serverAddress = new InetSocketAddress(address, 80);
        this.resource = resource;
    }

    public String getLimitedResponse() throws IOException {
        try ( var socketChannel = SocketChannel.open(serverAddress) ) {
            ByteBuffer sendBuffer = ByteBuffer.allocate(1024);
            ByteBuffer receiveBuffer = ByteBuffer.allocate(1024);

            HTTPReader reader = new HTTPReader(socketChannel, receiveBuffer);

            sendBuffer.put(ASCII.encode("GET " + resource + " HTTP/1.1\r\nHost: " + serverAddress.getHostName() + "\r\n\r\n"));
            sendBuffer.flip();
            socketChannel.write(sendBuffer);

            HTTPHeader responseHeader = reader.readHeader();
            System.out.println(responseHeader.toString());
            if ( responseHeader.isChunkedTransfer() ) {
                throw new IllegalStateException("This page is chuncked that is not supported yet.");
            } else if ( !responseHeader.getContentType().equals("text/html") ) {
                throw new IllegalStateException("The content type : " + responseHeader.getContentType() + " is not supported yet.");
            }

            var responseContentBuffer = reader.readBytes(responseHeader.getContentLength());
            var responseCharset = responseHeader.getCharset();
            if ( responseCharset == null ) {
                responseCharset = ISO;
            }

            responseContentBuffer.flip();
            System.out.println(responseCharset + " <= ");
            return responseCharset.decode(responseContentBuffer).toString();
        }
    }

    public static void main(String[] args) throws IOException {
        HTTPClient client = new HTTPClient("igm.univ-mlv.fr", "/~carayol/");
        System.out.println(client.getLimitedResponse());
    }
}
