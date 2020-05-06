package fr.upem.net.tcp.nonblocking;

import java.nio.ByteBuffer;

public class MessageReader implements Reader<Message> {

    private enum State {DONE,WAITING_PSEUDO,WAITING_MESSAGE,ERROR};

    private State state = State.WAITING_PSEUDO;
    private String pseudo;
    private Message value;
    private final StringReader stringReader = new StringReader();

    @Override
    public ProcessStatus process(ByteBuffer bb) {

        if ( state == State.DONE || state == State.ERROR ) {
            throw new IllegalStateException("Try to process on wrong state : " + state);
        }

        if ( state == State.WAITING_PSEUDO ) {
            ProcessStatus ps = stringReader.process(bb);
            switch (ps) {
                case DONE:
                    pseudo = stringReader.get();

                    /* Reset String reader for read the message. */
                    stringReader.reset();
                    state = State.WAITING_MESSAGE;
                    break;
                case REFILL:
                    return ProcessStatus.REFILL;
                case ERROR:
                    return ProcessStatus.ERROR;
            }
        }

        ProcessStatus ps = stringReader.process(bb);
        switch (ps) {
            case DONE:
                value = new Message(pseudo, stringReader.get());
                state = State.DONE;
                break;
            case REFILL:
                return ProcessStatus.REFILL;
            case ERROR:
                return ProcessStatus.ERROR;
        }

        return ProcessStatus.DONE;
    }

    @Override
    public Message get() {
        if ( state != State.DONE ) {
            throw new IllegalStateException("Try to get on wrong state : " + state);
        }

        return value;
    }

    @Override
    public void reset() {
        state = State.WAITING_PSEUDO;
        stringReader.reset();
        pseudo = null;
        value = null;
    }
}
