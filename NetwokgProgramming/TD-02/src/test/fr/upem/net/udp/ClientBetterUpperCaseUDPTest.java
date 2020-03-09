package fr.upem.net.udp;

import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.util.Optional;

import static fr.upem.net.udp.ClientBetterUpperCaseUDP.decodeMessage;
import static fr.upem.net.udp.ClientBetterUpperCaseUDP.encodeMessage;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertFalse;


public class ClientBetterUpperCaseUDPTest {

    private static ByteBuffer byteBufferFromHexaString(String content,int size){
        ByteBuffer bb = ByteBuffer.allocate(size);
        putHexaString(bb,content);
        bb.flip();
        return bb;
    }

    private static ByteBuffer byteBufferFromHexaString(String content){
        return byteBufferFromHexaString(content,content.length()/2);
    }


    // put the bytes described by the content string in hexa in the ByteBuffer
    private static void putHexaString(ByteBuffer bb,String content){
        while (!content.isEmpty()) {
            bb.put((byte) Integer.parseInt(content.substring(0, 2), 16));
            content = content.substring(2);
        }
    }

    // get the hexa string representing the ByteBuffer workspace
    private static String getHexaString(ByteBuffer bb){
        StringBuilder sb = new StringBuilder();
        while (bb.hasRemaining()) {
            sb.append(String.format("%02X",bb.get()));
         }
         return sb.toString();
    }

    @Test
    public void decodeMessageBasic1()  {
        ByteBuffer bb = byteBufferFromHexaString("000000066C6174696E31612424242121E9");
        bb.compact();
        Optional<String> res = ClientBetterUpperCaseUDP.decodeMessage(bb);
        assertTrue(res.isPresent());
        assertEquals("a$$$!!\u00e9", res.get());
    }

    @Test
    public void decodeMessageBasic2() {
        ByteBuffer bb = byteBufferFromHexaString("000000066C6174696E31612424242121E9", 100);
        bb.compact();
        Optional<String> res = ClientBetterUpperCaseUDP.decodeMessage(bb);
        assertTrue(res.isPresent());
        assertEquals("a$$$!!\u00e9", res.get());
    }

    @Test
    public void decodeMessageBasic3() {
        ByteBuffer bb = byteBufferFromHexaString("000000055554462D3861E282AC");
        bb.compact();
        Optional<String> res = ClientBetterUpperCaseUDP.decodeMessage(bb);
        assertTrue(res.isPresent());
        assertEquals("a\u20AC", res.get());
    }

    @Test
    public void decodeMessageWrongEncoding1() {
        ByteBuffer bb = byteBufferFromHexaString("000000066C6174696E31612424242121E9");
        bb.position(2);
        assertFalse(decodeMessage(bb).isPresent());
    }

    @Test
    public void decodeMessageWrongEncoding2() {
        ByteBuffer bb2 = byteBufferFromHexaString("FFFFFFFF6C6174696E31612424242121E9");
        bb2.compact();
        assertFalse(decodeMessage(bb2).isPresent());
    }

    @Test
    public void decodeMessageWrongEncoding3() {
        ByteBuffer bb3 = byteBufferFromHexaString("000000FF6C6174696E31612424242121E9");
        bb3.compact();
        assertFalse(decodeMessage(bb3).isPresent());
    }

    @Test
    public void decodeMessageWrongEncoding4() {
        ByteBuffer bb4 = byteBufferFromHexaString("00000006746174696E31612424242121E9");
        bb4.compact();
        assertFalse(decodeMessage(bb4).isPresent());
    }

    @Test
    public void encodeMessageBasic() throws Exception {
        Optional<ByteBuffer> bb = encodeMessage("a$$$!!\u00e9", "latin1");
        assertTrue(bb.isPresent());
        assertEquals("000000066C6174696E31612424242121E9", getHexaString(bb.get().flip()));
    }

    @Test
    public void encodeMessageBasic2() throws Exception {
        Optional<ByteBuffer> bb = encodeMessage("a\u20AC","UTF-8");
        assertEquals("000000055554462D3861E282AC",getHexaString(bb.get().flip()));

    }

    @Test
    public void encodeMessageTooLong() throws Exception {
        StringBuilder sb = new StringBuilder();
        for(int i=0;i<500;i++){
            sb.append("\u20AC");
        }
        Optional<ByteBuffer> bb = encodeMessage(sb.toString(),"UTF-8");
        assertFalse(bb.isPresent());

    }

}