package fr.upem.net.tcp.nonblocking;

import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;

public class StringReader implements Reader<String> {

    private enum State {DONE,WAITING_SIZE,WAITING_STRING,ERROR};

    private static final int BUFFER_SIZE = 1024;
    private State state = State.WAITING_SIZE;
    private final IntReader intReader = new IntReader();
    private final ByteBuffer internalbb = ByteBuffer.allocate(BUFFER_SIZE); // write-mode
    private String value;
    private static Charset UTF_8 = StandardCharsets.UTF_8;

    private void processGetSize(ByteBuffer byteBuffer) {
        ProcessStatus status = intReader.process(byteBuffer);
        switch (status){
            case DONE:
                var size = intReader.get();
                if ( size < 0 || size > BUFFER_SIZE ) {
                    state = State.ERROR;
                    return;
                }

                internalbb.limit(size);

                state = State.WAITING_STRING;
                break;
            case REFILL:
                return;
            case ERROR:
                state = State.ERROR;
                return;
        }
    }

    private void processGetString(ByteBuffer byteBuffer) {
        if ( byteBuffer.remaining() <= internalbb.remaining() ){
            internalbb.put(byteBuffer);
        } else {
            var oldLimit = byteBuffer.limit();
            byteBuffer.limit(internalbb.remaining());
            internalbb.put(byteBuffer);
            byteBuffer.limit(oldLimit);
        }
    }

    @Override
    public ProcessStatus process(ByteBuffer bb) {
        if ( state ==  State.DONE || state == State.ERROR ) {
            throw new IllegalStateException("Call to process but reader is in " + state + " mode.");
        }

        if ( state == State.WAITING_SIZE ) {
            processGetSize(bb);
        }

        bb.flip();
        try {
            if ( state == State.WAITING_STRING ) {
                processGetString(bb);
            }
        } finally {
            bb.compact();
        }

        if ( state == State.ERROR ) {
            return ProcessStatus.ERROR;
        }

        if (internalbb.hasRemaining()){
            return ProcessStatus.REFILL;
        }
        state = State.DONE;
        internalbb.flip();
        value = UTF_8.decode(internalbb).toString();
        return ProcessStatus.DONE;
    }

    @Override
    public String get() {
        if ( state != State.DONE ) {
            throw new IllegalStateException("Try to get on wrong state : " + state);
        }
        return value;
    }

    @Override
    public void reset() {
        state= State.WAITING_SIZE;
        internalbb.clear();
        intReader.reset();
        value = null;
    }
}
