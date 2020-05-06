package fr.upem.net.tcp.nonblocking;

import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;

public class Message {

    private static Charset UTF_8 = StandardCharsets.UTF_8;
    private static int BUFFER_SIZE = 2_048;

    private final String pseudo;

    private final String message;

    public Message(String pseudo, String message) {
        this.pseudo = pseudo;
        this.message = message;
    }

    public String getPseudo() {
        return pseudo;
    }

    public String getMessage() {
        return message;
    }

    /* In writemode :3 */
    public ByteBuffer getBufferized() {
        var byteBuffer = ByteBuffer.allocate(BUFFER_SIZE);
        var encodedPseudo = UTF_8.encode(pseudo);
        byteBuffer.putInt(encodedPseudo.remaining());
        byteBuffer.put(encodedPseudo);
        var encodedMessage = UTF_8.encode(message);
        byteBuffer.putInt(encodedMessage.remaining());
        byteBuffer.put(encodedMessage);
        return byteBuffer;
    }

}
