package fr.upem.net.buffers;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;
import java.nio.charset.Charset;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

public class ReadFileWithEncoding {

    private static void usage(){
        System.out.println("Usage: ReadFileWithEncoding charset filename");
    }

    private static String stringFromFile(Charset cs,Path path) throws IOException {
        try ( var fc = FileChannel.open(path) ) {
            var bb = ByteBuffer.allocate((int)fc.size());
            while ( fc.read(bb) > 0 );

            /* Why flip necessary ? */
            bb.flip();
            var cb = cs.decode(bb);
            return cb.toString();
        }
    }

    public static void main(String[] args) throws IOException {
        if (args.length!=2){
            usage();
            return;
        }
        Charset cs=Charset.forName(args[0]);
        Path path=Paths.get(args[1]);
        System.out.print(stringFromFile(cs,path));
    }


}
