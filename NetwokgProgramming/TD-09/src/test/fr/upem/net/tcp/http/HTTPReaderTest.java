package fr.upem.net.tcp.http;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.nio.ByteBuffer;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.channels.SocketChannel;
import java.nio.charset.StandardCharsets;


/**
 *
 * <p>
 * Tests suit for the class HTTPReader
 */
public class HTTPReaderTest {

    /**
     * Test for ReadLineLFCR with a null Socket
     */

    @Test
    public void testReadLineLFCR1() throws IOException {
        try {
            final String BUFFER_INITIAL_CONTENT = "Debut\rSuite\n\rFin\n\r\nANEPASTOUCHER";
            ByteBuffer buff = ByteBuffer.wrap(BUFFER_INITIAL_CONTENT.getBytes("ASCII")).compact();
            HTTPReader reader = new HTTPReader(null, buff);
            assertEquals("Debut\rSuite\n\rFin\n", reader.readLineCRLF());
            ByteBuffer buffFinal = ByteBuffer.wrap("ANEPASTOUCHER".getBytes("ASCII")).compact();
            assertEquals(buffFinal.flip(), buff.flip());
        } catch (NullPointerException e) {
            fail("The socket must not be read until buff is entirely consumed.");
        }
    }

    /**
     * Test for ReadLineLFCR with a fake server
     * @throws java.io.IOException
     */
    @Test
    public void testLineReaderLFCR2() throws IOException {
        FakeHTTPServer server = new FakeHTTPServer("Line1\r\nLine2\nLine2cont\r\n",7);
        try {
            server.serve();
            SocketChannel sc = SocketChannel.open();
            sc.connect(new InetSocketAddress("localhost", server.getPort()));
            var buff = ByteBuffer.allocate(12);
            buff.put("AA\r".getBytes("ASCII"));
            HTTPReader reader = new HTTPReader(sc, buff);
            assertEquals("AA\rLine1", reader.readLineCRLF());
            assertEquals("Line2\nLine2cont", reader.readLineCRLF());
        } finally {
            server.shutdown();
        }
    }


    /**
     * Test for ReadLineLFCR with a fake server closing the connection before the line is fully read
     * We expect an HTTPException as the server close the connection before sending a complete LFCR terminated
     * line.
     * @throws java.io.IOException
     */
    @Test
    public void testLineReaderLFCR3() throws IOException {
        FakeHTTPServer server = new FakeHTTPServer("Line1\nLine2\nLine2cont\r", 7);
        try {
            server.serve();
            SocketChannel sc = SocketChannel.open();
            sc.connect(new InetSocketAddress("localhost", server.getPort()));
            HTTPReader reader = new HTTPReader(sc, ByteBuffer.allocate(12));
            assertThrows(HTTPException.class, () -> reader.readLineCRLF());
        } finally {
            server.shutdown();
        }
    }

    /**
     * Test for ReadLineLFCR with a fake server
     * @throws java.io.IOException
     */
    @Test
    public void testLineReaderLFCR4() throws IOException {
        FakeHTTPServer server = new FakeHTTPServer("Line1\r\n€",10);
        server.serve();
        SocketChannel sc = SocketChannel.open();
        sc.connect(new InetSocketAddress("localhost", server.getPort()));
        ByteBuffer buff = ByteBuffer.allocate(12);
        HTTPReader reader = new HTTPReader(sc, buff);
        assertEquals("Line1", reader.readLineCRLF());
        ByteBuffer buffFinal = ByteBuffer.wrap("€".getBytes("UTF8")).compact();
        // Checks that the content of the buffer is the bytes of euro in UTF8
        buffFinal.flip();
        buff.flip();
        assertEquals(buffFinal.remaining(),buff.remaining());
        assertEquals(buffFinal.get(), buff.get());
        assertEquals(buffFinal.get(), buff.get());
        assertEquals(buffFinal.get(), buff.get());
        server.shutdown();
    }

    /**
     * Test for readBytes with a null SocketChannel
     * @throws java.io.IOException
     */
    @Test
    public void testReadBytes1() throws IOException {
        try {
            final String BUFFER_INITIAL_CONTENT = "1234567890ABCDEFGH";
            ByteBuffer buff = ByteBuffer.wrap(BUFFER_INITIAL_CONTENT.getBytes("ASCII")).compact();
            HTTPReader reader = new HTTPReader(null, buff);
            assertEquals("1234567890", StandardCharsets.UTF_8.decode(reader.readBytes(10).flip()).toString());
            ByteBuffer buffFinal = ByteBuffer.wrap("ABCDEFGH".getBytes("ASCII")).compact();
            assertEquals(buffFinal.flip(), buff.flip());
        } catch (NullPointerException e) {
            fail("The socket must not be read until buff is entirely consumed.");
        }
    }

    /**
     * Test for readBytes with FakeServer
     * @throws java.io.IOException
     */
    @Test
    public void testReadBytes2() throws IOException {
        FakeHTTPServer server = new FakeHTTPServer("DEFGH",4);
        try {
            server.serve();
            SocketChannel sc = SocketChannel.open();
            sc.connect(new InetSocketAddress("localhost", server.getPort()));
            var buff = ByteBuffer.allocate(12);
            buff.put("AA\r\nB".getBytes("ASCII"));
            HTTPReader reader = new HTTPReader(sc, buff);
            assertEquals("AA\r\nBDE", StandardCharsets.US_ASCII.decode(reader.readBytes(7).flip()).toString());
        } finally {
            server.shutdown();
        }
    }

    /**
     * Test for readBytes with FakeServer
     * @throws java.io.IOException
     */
    @Test
    public void testReadChunks() throws IOException {
        FakeHTTPServer server = new FakeHTTPServer("i\r\n5\r\npedia\r\nE\r\n in\r\n\r\nchunks.\r\n0\r\n\r\n",4);
        try {
            server.serve();
            SocketChannel sc = SocketChannel.open();
            sc.connect(new InetSocketAddress("localhost", server.getPort()));
            var buff = ByteBuffer.allocate(12);
            buff.put("4\r\nWik".getBytes("ASCII"));
            HTTPReader reader = new HTTPReader(sc, buff);
            assertEquals("Wikipedia in\r\n\r\nchunks.", StandardCharsets.US_ASCII.decode(reader.readChunks().flip()).toString());
        } finally {
            server.shutdown();
        }
    }
}
