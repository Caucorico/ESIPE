package fr.upem.net.tcp.nonblocking;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.*;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ServerEchoWithConsole {

	static private class Context {

		final private SelectionKey key;
		final private SocketChannel sc;
		final private ByteBuffer bb = ByteBuffer.allocate(BUFFER_SIZE);
		private boolean closed = false;

		private Context(SelectionKey key){
			this.key = key;
			this.sc = (SocketChannel) key.channel();
		}

		/**
		 * Update the interestOps of the key looking
		 * only at values of the boolean closed and
		 * the ByteBuffer buff.
		 *
		 * The convention is that buff is in write-mode.
		 */
		private void updateInterestOps() {
			var interestOps = 0x0;

			if ( !closed && bb.hasRemaining() ) {
				interestOps |= SelectionKey.OP_READ;

			}

			if ( bb.position() > 0 ) {
				interestOps |= SelectionKey.OP_WRITE;
			}

			if ( interestOps == 0 ) {
				silentlyClose();
				return;
			}

			key.interestOps(interestOps);
		}

		/**
		 * Performs the read action on sc
		 *
		 * The convention is that buff is in write-mode before calling doRead
		 * and is in write-mode after calling doRead
		 *
		 * @throws IOException
		 */
		private void doRead() throws IOException {
			var res = sc.read(bb);

			if ( res == -1 ) {
				logger.info("Connection closed with the client.");
				closed = true;
			}

			updateInterestOps();
		}

		/**
		 * Performs the write action on sc
		 *
		 * The convention is that buff is in write-mode before calling doWrite
		 * and is in write-mode after calling doWrite
		 *
		 * @throws IOException
		 */
		private void doWrite() throws IOException {
			bb.flip();
			sc.write(bb);
			bb.compact();

			updateInterestOps();
		}

		private void closeReading() {
			closed = true;
			updateInterestOps();
		}

		private void silentlyClose() {
			try {
				sc.close();
			} catch (IOException e) {
				// ignore exception
			}
		}
	}

	static private int BUFFER_SIZE = 1_024;
	static private Logger logger = Logger.getLogger(ServerEchoWithConsole.class.getName());

	private final ServerSocketChannel serverSocketChannel;
	private final Selector selector;
	private Thread serverThread;
	private volatile boolean serverClosed = false;

	private final Object lock = new Object();
	private long clientNumber = 0;

	public ServerEchoWithConsole(int port) throws IOException {
		serverSocketChannel = ServerSocketChannel.open();
		serverSocketChannel.bind(new InetSocketAddress(port));
		selector = Selector.open();
	}

	private void interruptServer() {
		serverThread.interrupt();
	}

	private void closeAllConnection() {
		for ( var key : selector.keys() ) {
			silentlyClose(key);
		}
	}

	private void closeServer() {
		for ( var key : selector.keys() ) {
			if ( key.interestOps() == SelectionKey.OP_ACCEPT ) {
				silentlyClose(key);
			} else if ( (key.interestOps()&SelectionKey.OP_READ) != 0 || (key.interestOps() & SelectionKey.OP_WRITE) != 0 ) {
				var context = (Context) key.attachment();
				context.closeReading();
			}
		}
	}

	private void listenStdin() {
		try {
			var scan = new Scanner(System.in);
			while (scan.hasNextLine()) {
				var msg = scan.nextLine();

				switch (msg) {
					case "INFO":
						selector.wakeup();
						synchronized (lock) {
							System.out.println("Client number : " + clientNumber);
						}
						break;
					case "SHUTDOWN":
						/* serverClosed volatile */
						serverClosed = true;
						selector.wakeup();
						System.out.println("SHUTDOWN order received.");
						break;
					case "SHUTDOWNNOW":
						System.out.println("SHUTDOWNNOW order received.");
						interruptServer();
						return;
				}
			}
		}finally {
			logger.info("Console thread stopping");
		}
	}

	public void launch() throws InterruptedException {
		serverThread = new Thread( () -> {
			try {
				serverSocketChannel.configureBlocking(false);
				serverSocketChannel.register(selector, SelectionKey.OP_ACCEPT);
			} catch (IOException e) {
				logger.log(Level.SEVERE, "Error during the configuration of the socketChannel", e);
				return;
			}

			while(!Thread.interrupted()) {
				printKeys(); // for debug
				System.out.println("Starting select");
				try {
					selector.select(this::treatKey);
				} catch (UncheckedIOException | IOException e) {
					logger.log(Level.SEVERE, "Error during the select of the socketChannel", e);
					return;
				}
				System.out.println("Select finished");

				System.out.println("Update client size");

				synchronized (lock) {
					clientNumber = 0;

					for ( var key : selector.keys() ) {
						if ( key.interestOps() != SelectionKey.OP_ACCEPT ) {
							clientNumber++;
						}
					}
				}

				System.out.println("Update client size finished");

				/* server closed volatile */
				if ( serverClosed ) {
					closeServer();
				}
			}
			closeAllConnection();
			System.out.println("SHUTDOWNNOW finished.");
		});
		serverThread.start();
		listenStdin();
		serverThread.join();
	}

	private void treatKey(SelectionKey key) {
		printSelectedKey(key); // for debug
		try {
			if (key.isValid() && key.isAcceptable()) {
				doAccept(key);
			}
		} catch(IOException ioe) {
			// lambda call in select requires to tunnel IOException
			throw new UncheckedIOException(ioe);
		}
		try {
			if (key.isValid() && key.isWritable()) {
				((Context) key.attachment()).doWrite();
			}
			if (key.isValid() && key.isReadable()) {
				((Context) key.attachment()).doRead();
			}
		} catch (IOException e) {
			logger.log(Level.INFO,"Connection closed with client due to IOException",e);
			silentlyClose(key);
		}
	}

	private void doAccept(SelectionKey key) throws IOException {
		SocketChannel sc = serverSocketChannel.accept();
		if ( sc == null ) {
			logger.log(Level.WARNING, "Bad hint");
			return;
		}

		sc.configureBlocking(false);
		var sk =  sc.register(selector, SelectionKey.OP_READ);
		var context = new Context(sk);
		sk.attach(context);
	}

	private void silentlyClose(SelectionKey key) {
		Channel sc = (Channel) key.channel();
		try {
			sc.close();
		} catch (IOException e) {
			// ignore exception
		}
	}

	public static void main(String[] args) throws NumberFormatException, IOException, InterruptedException {
		if (args.length!=1){
			usage();
			return;
		}
		new ServerEchoWithConsole(Integer.parseInt(args[0])).launch();
	}

	private static void usage(){
		System.out.println("Usage : ServerEcho port");
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
