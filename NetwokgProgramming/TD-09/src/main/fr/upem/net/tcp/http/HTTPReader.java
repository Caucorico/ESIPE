package fr.upem.net.tcp.http;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.SocketChannel;
import java.nio.charset.Charset;
import java.util.HashMap;
import java.util.Map;


public class HTTPReader {

    private final Charset ASCII_CHARSET = Charset.forName("ASCII");
    private final SocketChannel sc;
    private final ByteBuffer buff;

    public HTTPReader(SocketChannel sc, ByteBuffer buff) {
        this.sc = sc;
        this.buff = buff;
    }

    /**
     * @return The ASCII string terminated by CRLF without the CRLF
     * <p>
     * The method assume that buff is in write mode and leave it in write-mode
     * The method never reads from the socket as long as the buffer is not empty
     * @throws IOException HTTPException if the connection is closed before a line could be read
     */
    public String readLineCRLF() throws IOException {
        char a = '\0', b = '\0';
        StringBuilder sb = new StringBuilder();

        buff.flip();

        do {
            if ( !buff.hasRemaining() ) {
                buff.clear();

                var res = sc.read(buff);
                if (res == -1 ) {
                    throw new HTTPException();
                }
                buff.flip();
            }

            a = b;
            b = (char)buff.get();
            sb.append(b);
        } while ( a != '\r' || b != '\n');

        buff.compact();

        sb.setLength(sb.length()-2);

        return sb.toString();
    }

    /**
     * @return The HTTPHeader object corresponding to the header read
     * @throws IOException HTTPException if the connection is closed before a header could be read
     *                     if the header is ill-formed
     */
    public HTTPHeader readHeader() throws IOException {
        Map<String, String> params = new HashMap<>();
        String firstLine = null;

        while ( true ) {
            var line = readLineCRLF();
            if ( line.length() == 0 ) {
                break;
            }

            if ( firstLine == null ) {
                firstLine = line;
                continue;
            }

            String[] split = line.split(":");
            if ( split.length < 2 ) {
                throw new HTTPException();
            }

            params.computeIfPresent(split[0], (k, v) -> v+";"+split[1]);
            params.putIfAbsent(split[0], split[1]);
        }

        return HTTPHeader.create(firstLine, params);
    }

    /**
     * @param size
     * @return a ByteBuffer in write-mode containing size bytes read on the socket
     * @throws IOException HTTPException is the connection is closed before all bytes could be read
     */
    public ByteBuffer readBytes(int size) throws IOException {
        ByteBuffer newByteBuffer = ByteBuffer.allocate(size);
        /* TODO : Ask to the teacher why the read can read more than the response-content size */
        var oldLimit = buff.limit();
        buff.limit(Math.min(buff.capacity(), size));
        buff.flip();
        newByteBuffer.put(buff);
        buff.limit(oldLimit);
        buff.compact();

        while ( newByteBuffer.hasRemaining() ) {
            var res = sc.read(newByteBuffer);
            if ( res == -1 ) {
                throw new HTTPException();
            }
        }

        return newByteBuffer;
    }

    private static boolean readFully(ByteBuffer byteBuffer, SocketChannel socketChannel) throws IOException {
        while (byteBuffer.hasRemaining()) {
            var res = socketChannel.read(byteBuffer);
            if ( res == -1 ) {
                throw new HTTPException();
            }
        }

        return true;
    }

    /**
     * @return a ByteBuffer in write-mode containing a content read in chunks mode
     * @throws IOException HTTPException if the connection is closed before the end of the chunks
     *                     if chunks are ill-formed
     */
    public ByteBuffer readChunks() throws IOException {
        ByteBuffer byteBuffer = ByteBuffer.allocate(1024);

        while ( true ) {
            String line = readLineCRLF();

            if ( line.length() == 0 ) {
                break;
            }

            System.out.println(line);
            System.out.println("##########################################");

            int chunkSize = Integer.parseInt(line, 16) + 2;

            if ( byteBuffer.remaining() < chunkSize ) {
                var newByteBuffer = ByteBuffer.allocate(byteBuffer.capacity()+chunkSize);
                byteBuffer.flip();
                newByteBuffer.put(byteBuffer);
                byteBuffer = newByteBuffer;
            }

            var nbb = readBytes(chunkSize);
            nbb.flip();
            byteBuffer.put(nbb);
            System.out.println(byteBuffer.get());
            System.out.println(byteBuffer.get());
            byteBuffer.position(byteBuffer.position());
        }

        return byteBuffer;
    }


    public static void main(String[] args) throws IOException {
        Charset charsetASCII = Charset.forName("ASCII");
        String request = "GET / HTTP/1.1\r\n"
                + "Host: www.w3.org\r\n"
                + "\r\n";
        SocketChannel sc = SocketChannel.open();
        sc.connect(new InetSocketAddress("www.w3.org", 80));
        sc.write(charsetASCII.encode(request));
        ByteBuffer bb = ByteBuffer.allocate(50);
        HTTPReader reader = new HTTPReader(sc, bb);
        System.out.println(reader.readLineCRLF());
        System.out.println(reader.readLineCRLF());
        System.out.println(reader.readLineCRLF());
        sc.close();

        bb = ByteBuffer.allocate(50);
        sc = SocketChannel.open();
        sc.connect(new InetSocketAddress("www.w3.org", 80));
        reader = new HTTPReader(sc, bb);
        sc.write(charsetASCII.encode(request));
        System.out.println(reader.readHeader());
        sc.close();

        bb = ByteBuffer.allocate(50);
        sc = SocketChannel.open();
        sc.connect(new InetSocketAddress("www.w3.org", 80));
        reader = new HTTPReader(sc, bb);
        sc.write(charsetASCII.encode(request));
        HTTPHeader header = reader.readHeader();
        System.out.println(header);
        ByteBuffer content = reader.readBytes(header.getContentLength());
        content.flip();
        System.out.println(header.getCharset().decode(content));
        sc.close();

        bb = ByteBuffer.allocate(50);
        request = "GET / HTTP/1.1\r\n"
                + "Host: www.u-pem.fr\r\n"
                + "\r\n";
        sc = SocketChannel.open();
        sc.connect(new InetSocketAddress("www.u-pem.fr", 80));
        reader = new HTTPReader(sc, bb);
        sc.write(charsetASCII.encode(request));
        header = reader.readHeader();
        System.out.println(header);
        content = reader.readChunks();
        content.flip();
        System.out.println(header.getCharset().decode(content));
        sc.close();
    }
}
