package fr.upem.net.tcp;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.SocketChannel;
import java.util.*;
import java.util.logging.Logger;

public class ClientLongSum {

    private static final int BUFFER_SIZE = 1024;
    public static final Logger logger = Logger.getLogger(ClientLongSum.class.getName());

    private static ArrayList<Long> randomLongList(int size){
        Random rng= new Random();
        ArrayList<Long> list= new ArrayList<>(size);
        for(int i=0;i<size;i++){
            list.add(rng.nextLong());
        }
        return list;
    }

    private static boolean checkSum(List<Long> list, long response) {
        long sum = 0;
        for(long l : list)
            sum += l;
        return sum==response;
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
        /* TODO : Ask to the teacher why my function doesn't return true ;) */

        do {
            var res = sc.read(bb);
            if ( res == -1 ) return false;
        } while (bb.hasRemaining());

        return false;
    }


    /**
     * Write all the longs in list in BigEndian on the server
     * and read the long sent by the server and returns it
     *
     * returns Optional.empty if the protocol is not followed by the server but no IOException is thrown
     *
     * @param sc
     * @param list
     * @return
     * @throws IOException
     */
    private static Optional<Long> requestSumForList(SocketChannel sc, List<Long> list) throws IOException {
        ByteBuffer byteBuffer = ByteBuffer.allocate(BUFFER_SIZE);
        Iterator<Long> iterator = list.iterator();

        byteBuffer.putInt(list.size());

        while ( iterator.hasNext() ) {
            while ( byteBuffer.remaining() > Long.BYTES ) {
                byteBuffer.putLong(iterator.next());
                if ( !iterator.hasNext() ) break;
            }
            byteBuffer.flip();
            sc.write(byteBuffer);
            if ( byteBuffer.hasRemaining() ) return Optional.empty();
            byteBuffer.clear();
        }

        byteBuffer.clear();
        byteBuffer.limit(Long.BYTES);
        while ( readFully(sc, byteBuffer) );
        if ( byteBuffer.hasRemaining() ) return Optional.empty();
        byteBuffer.flip();

        return Optional.of(byteBuffer.getLong());
    }

    public static void main(String[] args) throws IOException {
        InetSocketAddress server = new InetSocketAddress(args[0],Integer.valueOf(args[1]));
        try (SocketChannel sc = SocketChannel.open(server)) {
            for(int i=0; i<5; i++) {
                ArrayList<Long> list = randomLongList(50);

                Optional<Long> l = requestSumForList(sc, list);
                if (!l.isPresent()) {
                    System.err.println("Connection with server lost.");
                    return;
                }
                if (!checkSum(list, l.get())) {
                    System.err.println("Oups! Something wrong happens!");
                }
            }
            System.err.println("Everything seems ok");
        }
    }
}
