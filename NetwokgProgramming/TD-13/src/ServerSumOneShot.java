import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.*;
import java.util.ArrayList;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ServerSumOneShot {

	static private int BUFFER_SIZE = 2*Integer.BYTES;
	static private Logger logger = Logger.getLogger(ServerSumOneShot.class.getName());

	private final ServerSocketChannel serverSocketChannel;
	private final Selector selector;

	public ServerSumOneShot(int port) throws IOException {
		serverSocketChannel = ServerSocketChannel.open();
		serverSocketChannel.bind(new InetSocketAddress(port));
		selector = Selector.open();
	}

	public void launch() throws IOException {
		serverSocketChannel.configureBlocking(false);
		serverSocketChannel.register(selector, SelectionKey.OP_ACCEPT);
		while (!Thread.interrupted()) {
			printKeys(); // for debug
			System.out.println("Starting select");
			try {
				selector.select(this::treatKey);
			} catch (UncheckedIOException e) {
				throw e.getCause();
			}

			System.out.println("Select finished");
		}
	}

	private void treatKey(SelectionKey key) {
		printSelectedKey(key); // for debug
		if (key.isValid() && key.isAcceptable()) {
			try {
				doAccept(key);
			} catch (IOException e) {
				logger.log(Level.SEVERE, "Error during the accept", e);
				throw new UncheckedIOException(e);
			}
		}
		try {
			if (key.isValid() && key.isWritable()) {
				doWrite(key);
			}
			if (key.isValid() && key.isReadable()) {
				doRead(key);
			}
		} catch (AsynchronousCloseException e) {
			logger.log(Level.INFO, "Client close the connection");
			silentlyClose(key);
			/* TODO : Ask to the teacher if close the connection remove the key from the list */
		} catch (IOException e) {
			logger.log(Level.SEVERE, "Error during the IO", e);
			throw new UncheckedIOException(e);
		}
	}

	private void doAccept(SelectionKey key) throws IOException {
		ServerSocketChannel ssc = (ServerSocketChannel) key.channel();
		SocketChannel sc = ssc.accept();
		if ( sc == null ) {
			/* TODO : ask to the teacher if warning ? */
			logger.log(Level.WARNING, "Bad hint");
			return;
		}

		sc.configureBlocking(false);
		sc.register(selector, SelectionKey.OP_READ, ByteBuffer.allocate(BUFFER_SIZE));
	}

	private void doRead(SelectionKey key) throws IOException {
		var socketChannel = (SocketChannel) key.channel();
		var byteBuffer = (ByteBuffer)key.attachment();
		int result = socketChannel.read(byteBuffer);

		if (result == -1) {
			logger.info("Connection closed by client.");
			silentlyClose(key);
			return;
		}

		if ( !byteBuffer.hasRemaining() ) {
			key.interestOps(SelectionKey.OP_WRITE);
			byteBuffer.flip();
		}
	}

	/**
	 *
	 * @param socketChannel
	 * @param byteBuffer ByteBuffer in read mode.
	 * @return
	 * @throws IOException
	 */
	private static boolean writeFully(SocketChannel socketChannel, ByteBuffer byteBuffer) throws IOException {
		socketChannel.write(byteBuffer);
		return !byteBuffer.hasRemaining();
	}

	private void doWrite(SelectionKey key) throws IOException {
		var byteBuffer = (ByteBuffer)key.attachment();
		var socketChannel = (SocketChannel)key.channel();
		var integer1 = byteBuffer.getInt();
		var integer2 = byteBuffer.getInt();

		var sum = integer1 + integer2;

		byteBuffer.clear();
		byteBuffer.putInt(sum);
		byteBuffer.flip();
		if ( writeFully(socketChannel, byteBuffer) ) {
			/* TODO : close socket channel and remove it from keys */
			// ex 1 : silentlyClose(key);

			key.interestOps(SelectionKey.OP_READ);
			byteBuffer.clear();
		}
	}

	private void silentlyClose(SelectionKey key) {
		Channel sc = (Channel) key.channel();
		try {
			sc.close();
		} catch (IOException e) {
			// ignore exception
		}
	}

	public static void main(String[] args) throws NumberFormatException, IOException {
		if (args.length!=1){
			usage();
			return;
		}
		new ServerSumOneShot(Integer.parseInt(args[0])).launch();
	}

	private static void usage(){
		System.out.println("Usage : ServerSumOneShot port");
	}

	/***
	 *  Theses methods are here to help understanding the behavior of the selector
	 ***/

	private String interestOpsToString(SelectionKey key){
		if (!key.isValid()) {
			return "CANCELLED";
		}
		int interestOps = key.interestOps();
		ArrayList<String> list = new ArrayList<>();
		if ((interestOps&SelectionKey.OP_ACCEPT)!=0) list.add("OP_ACCEPT");
		if ((interestOps&SelectionKey.OP_READ)!=0) list.add("OP_READ");
		if ((interestOps&SelectionKey.OP_WRITE)!=0) list.add("OP_WRITE");
		return String.join("|",list);
	}

	public void printKeys() {
		Set<SelectionKey> selectionKeySet = selector.keys();
		if (selectionKeySet.isEmpty()) {
			System.out.println("The selector contains no key : this should not happen!");
			return;
		}
		System.out.println("The selector contains:");
		for (SelectionKey key : selectionKeySet){
			SelectableChannel channel = key.channel();
			if (channel instanceof ServerSocketChannel) {
				System.out.println("\tKey for ServerSocketChannel : "+ interestOpsToString(key));
			} else {
				SocketChannel sc = (SocketChannel) channel;
				System.out.println("\tKey for Client "+ remoteAddressToString(sc) +" : "+ interestOpsToString(key));
			}
		}
	}

	private String remoteAddressToString(SocketChannel sc) {
		try {
			return sc.getRemoteAddress().toString();
		} catch (IOException e){
			return "???";
		}
	}

	public void printSelectedKey(SelectionKey key) {
		SelectableChannel channel = key.channel();
		if (channel instanceof ServerSocketChannel) {
			System.out.println("\tServerSocketChannel can perform : " + possibleActionsToString(key));
		} else {
			SocketChannel sc = (SocketChannel) channel;
			System.out.println("\tClient " + remoteAddressToString(sc) + " can perform : " + possibleActionsToString(key));
		}
	}

	private String possibleActionsToString(SelectionKey key) {
		if (!key.isValid()) {
			return "CANCELLED";
		}
		ArrayList<String> list = new ArrayList<>();
		if (key.isAcceptable()) list.add("ACCEPT");
		if (key.isReadable()) list.add("READ");
		if (key.isWritable()) list.add("WRITE");
		return String.join(" and ",list);
	}
}
