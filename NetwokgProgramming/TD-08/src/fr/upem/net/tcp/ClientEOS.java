package fr.upem.net.tcp;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.SocketChannel;
import java.nio.charset.Charset;
import java.util.LinkedList;
import java.util.logging.Logger;

public class ClientEOS {

    public static final Charset UTF8_CHARSET = Charset.forName("UTF-8");
    public static final int BUFFER_SIZE = 1024;
    public static final Logger logger = Logger.getLogger(ClientEOS.class.getName());

    /**
     * This method:
     *  - connect to server
     *  - writes the bytes corresponding to request in UTF8
     *  - closes the write-channel to the server
     *  - stores the bufferSize first bytes of server response
     *  - return the corresponding string in UTF8
     *
     * @param request
     * @param server
     * @param bufferSize
     * @return the UTF8 string corresponding to bufferSize first bytes of server response
     * @throws IOException
     */

    public static String getFixedSizeResponse(String request, SocketAddress server, int bufferSize)
            throws IOException {

    	try (var socketChannel = SocketChannel.open(server)) {
			var byteBuffer = ByteBuffer.allocate(bufferSize);
			byteBuffer.put(UTF8_CHARSET.encode(request));
			byteBuffer.flip();
			socketChannel.write(byteBuffer);
			socketChannel.shutdownOutput();

			byteBuffer.clear();
			socketChannel.read(byteBuffer);
			byteBuffer.flip();

			return UTF8_CHARSET.decode(byteBuffer).toString();
		}
    }

    /**
  	 * This method:
	   *  - connect to server
	   *  - writes the bytes corresponding to request in UTF8
	   *  - closes the write-channel to the server
	   *  - reads and stores all bytes from server until read-channel is closed
	   *  - return the corresponding string in UTF8
	   *
	   * @param request
	   * @param server
	   * @return the UTF8 string corresponding the full response of the server
	   * @throws IOException
     */

    public static String getUnboundedResponse(String request, SocketAddress server) throws IOException {

    	try (var socketChannel = SocketChannel.open(server) ) {
			ByteBuffer byteBuffer = ByteBuffer.allocate(BUFFER_SIZE);
			byteBuffer.put(UTF8_CHARSET.encode(request));
			byteBuffer.flip();

			socketChannel.write(byteBuffer);
			socketChannel.shutdownOutput();

			byteBuffer.clear();

			while (readFully(socketChannel, byteBuffer)) {
				if ( byteBuffer.limit() <= byteBuffer.position() ) {
					var newByteBuffer = ByteBuffer.allocate(byteBuffer.capacity()*2);
					byteBuffer.flip();
					newByteBuffer.put(byteBuffer);
					byteBuffer = newByteBuffer;
				}
			}
			byteBuffer.flip();
			return UTF8_CHARSET.decode(byteBuffer).toString();
		}
    }

   /**
	  * Fill the workspace of the Bytebuffer with bytes read from sc.
	  *
	  * @param sc
	  * @param bb
	  * @return false if read returned -1 at some point and true otherwise
	  * @throws IOException
	  */
   static boolean readFully(SocketChannel sc, ByteBuffer bb) throws IOException {
		/* todo : ask to the teacher if the bytes that cannot be place in the buffer are lost  */

		do {
			var res = sc.read(bb);
			if ( res == -1 ) {
				return false;
			}
		} while ( bb.hasRemaining() );
	   return true;
   }

    public static void main(String[] args) throws IOException {
		  InetSocketAddress google = new InetSocketAddress("www.google.fr", 80);
//		  System.out.println(getFixedSizeResponse("GET / HTTP/1.1\r\nHost: www.google.fr\r\n\r\n",
//		  		google,	512));
		  System.out.println(getUnboundedResponse("GET / HTTP/1.1\r\nHost: www.google.fr\r\n\r\n",
		        google));
    }
}
