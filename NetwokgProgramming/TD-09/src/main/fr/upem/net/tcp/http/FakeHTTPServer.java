package fr.upem.net.tcp.http;

import java.io.IOException;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;

public class FakeHTTPServer {

    private final ServerSocketChannel ss;
    private final int port;
    private final ByteBuffer content;
    private final Thread t;


    public FakeHTTPServer(String s, int max) throws IOException {
        ss = ServerSocketChannel.open();
        ss.bind(null);
        InetSocketAddress address = (InetSocketAddress) ss.getLocalAddress();
        port = address.getPort();
        content = ByteBuffer.wrap(s.getBytes("UTF-8"));
        this.t = new Thread(() ->
        {
            SocketChannel sc = null;
            try {
                sc = ss.accept();
                while (!Thread.interrupted() && content.hasRemaining()) {
                    var oldlimit = content.limit();
                    content.limit(Math.min(content.position() + max, oldlimit));
                    sc.write(content);
                    Thread.sleep(100);
                    content.limit(oldlimit);
                }
            } catch (Exception e) {
                //
            } finally {
                try {
                    if (sc != null) sc.close();
                    ss.close();
                } catch (Exception e) {
                    //
                }
            }
        });
    }

    public FakeHTTPServer(InputStream in) throws IOException {
        ss = ServerSocketChannel.open();
        ss.bind(null);
        InetSocketAddress address = (InetSocketAddress) ss.getLocalAddress();
        port = address.getPort();
        content=null;
        this.t = new Thread(() ->
        {
            SocketChannel sc = null;
            try {
                sc = ss.accept();
                byte[] buff = new byte[100];
                ByteBuffer bb = ByteBuffer.wrap(buff);
                int read=0;
                while (!Thread.interrupted()&& (read=in.read(buff))!=-1){
                    bb.clear();
                    bb.limit(read);
                    sc.write(bb);
                    Thread.sleep(100);
                }
            } catch (Exception e) {
                //
            } finally {
                try {
                    if (sc != null) sc.close();
                    ss.close();
                } catch (Exception e) {
                    //
                }
            }
        });
    }

    public void serve() {
        t.start();
    }

    public void shutdown() {
        t.interrupt();
    }



    public int getPort() {
        return port;
    }
}

