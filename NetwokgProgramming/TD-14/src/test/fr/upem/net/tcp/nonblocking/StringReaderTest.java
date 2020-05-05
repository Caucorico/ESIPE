package fr.upem.net.tcp.nonblocking;

import fr.upem.net.tcp.nonblocking.Reader;
import fr.upem.net.tcp.nonblocking.StringReader;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

import static fr.upem.net.tcp.nonblocking.Reader.ProcessStatus.REFILL;
import static org.junit.jupiter.api.Assertions.*;

public class StringReaderTest {

    @Test
    public void simple(){
        var string = "\u20ACa\u20AC";
        var bb = ByteBuffer.allocate(1024);
        var bytes = StandardCharsets.UTF_8.encode(string);
        bb.putInt(bytes.remaining()).put(bytes);
        StringReader sr = new StringReader();
        assertEquals(Reader.ProcessStatus.DONE,sr.process(bb));
        assertEquals(string,sr.get());
        assertEquals(0,bb.position());
        assertEquals(bb.capacity(),bb.limit());
    }

    @Test
    public void reset(){
        var string = "\u20ACa\u20AC";
        var string2 = "\u20ACa\u20ACabcd";
        var bb = ByteBuffer.allocate(1024);
        var bytes = StandardCharsets.UTF_8.encode(string);
        var bytes2 = StandardCharsets.UTF_8.encode(string2);
        bb.putInt(bytes.remaining()).put(bytes).putInt(bytes2.remaining()).put(bytes2);
        StringReader sr = new StringReader();
        assertEquals(Reader.ProcessStatus.DONE,sr.process(bb));
        assertEquals(string,sr.get());
        assertEquals(15,bb.position());
        assertEquals(bb.capacity(),bb.limit());
        sr.reset();
        assertEquals(Reader.ProcessStatus.DONE,sr.process(bb));
        assertEquals(string2,sr.get());
        assertEquals(0,bb.position());
        assertEquals(bb.capacity(),bb.limit());
    }

    @Test
    public void smallBuffer(){
        var string = "\u20ACa\u20AC";
        var bb = ByteBuffer.allocate(1024);
        var bytes = StandardCharsets.UTF_8.encode(string);
        bb.putInt(bytes.remaining()).put(bytes).flip();
        var bbSmall = ByteBuffer.allocate(2);
        var sr = new StringReader();
        while (bb.hasRemaining()) {
            while(bb.hasRemaining() && bbSmall.hasRemaining()){
                bbSmall.put(bb.get());
            }
            if (bb.hasRemaining()) {
                assertEquals(Reader.ProcessStatus.REFILL,sr.process(bbSmall));
            } else {
                assertEquals(Reader.ProcessStatus.DONE,sr.process(bbSmall));
            }
        }
        assertEquals(string,sr.get());
    }

    @Test
    public void errorGet(){
        var sr = new StringReader();
        assertThrows(IllegalStateException.class,() -> {var res=sr.get();});
    }

    @Test
    public void errorNeg(){
        var sr = new StringReader();
        var bb = ByteBuffer.allocate(1024);
        var bytes = StandardCharsets.UTF_8.encode("aaaaa");
        bb.putInt(-1).put(bytes);
        assertEquals(Reader.ProcessStatus.ERROR,sr.process(bb));
    }

    @Test
    public void errorTooBig(){
        var sr = new StringReader();
        var bb = ByteBuffer.allocate(1024);
        var bytes = StandardCharsets.UTF_8.encode("aaaaa");
        bb.putInt(1025).put(bytes);
        assertEquals(Reader.ProcessStatus.ERROR,sr.process(bb));
    }
}
