package fr.upem.net.tcp.http;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.nio.channels.SocketChannel;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.logging.Logger;

public class HTTPClient {

    private final static Charset ASCII = StandardCharsets.US_ASCII;
    private final static Charset UTF8 = StandardCharsets.UTF_8;

    private final static Logger LOGGER = Logger.getLogger(HTTPClient.class.getName());

    private HTTPReader httpReader;

    private InetSocketAddress serverAddress;

    private static final int INIT_BUFFER_SIZE = 1024;
    private final ByteBuffer mainByteBuffer;

    public HTTPClient() throws IOException {
        this.mainByteBuffer = ByteBuffer.allocate(INIT_BUFFER_SIZE);
    }

    private void sendRequest(SocketChannel socketChannel, String resource) throws IOException {
        mainByteBuffer.clear();
        mainByteBuffer.put(ASCII.encode("GET " + resource + " HTTP/1.1\r\nHost: " + serverAddress.getHostName() + "\r\n\r\n"));
        mainByteBuffer.flip();
        socketChannel.write(mainByteBuffer);
    }

    private HTTPHeader getHTTPHeader(SocketChannel socketChannel, String resource) throws IOException {
        sendRequest(socketChannel, resource);
        mainByteBuffer.clear();
        return httpReader.readHeader();
    }

    private String getLimitedResponse(HTTPHeader httpResponseHeader) throws IOException {
        LOGGER.info("limited response");
        var buffer = httpReader.readBytes(httpResponseHeader.getContentLength());
        var charset = httpResponseHeader.getCharset();
        if ( charset == null ) charset = UTF8;
        buffer.flip();
        return charset.decode(buffer).toString();
    }

    private String getChunckedResponse(HTTPHeader httpResponseHeader) throws IOException {
        LOGGER.info("chuncked response");
        var buffer = httpReader.readChunks();
        var charset = httpResponseHeader.getCharset();
        if ( charset == null ) charset = UTF8;
        buffer.flip();
        return charset.decode(buffer).toString();
    }

    public String getResponse(String address, String resource) throws IOException {
        this.serverAddress = new InetSocketAddress(address, 80);
        try ( var socketChannel = SocketChannel.open(serverAddress) ) {
            this.httpReader = new HTTPReader(socketChannel, mainByteBuffer);
            var header = getHTTPHeader(socketChannel, resource);
            if ( header.getCode() == 301 || header.getCode() == 302 ) {
                var fields = header.getFields();
                URI uri;
                try {
                    uri = new URI(fields.get("location"));
                } catch (URISyntaxException e) {
                    throw new HTTPException();
                }
                return getResponse(uri.getHost(), uri.getPath());
            }

            if ( header.isChunkedTransfer() ) {
                return getChunckedResponse(header);
            } else {
                return getLimitedResponse(header);
            }
        }
    }

    public static void main(String[] args) throws IOException {
        HTTPClient client = new HTTPClient();
        System.out.println(client.getResponse("www.u-pem.fr", "/"));
    }
}
