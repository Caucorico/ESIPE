package fr.uge.poo.cmdline.ex6;

import org.junit.jupiter.api.Test;

import java.text.ParseException;
import java.util.BitSet;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;


@SuppressWarnings("static-method")
class CmdLineParserTest {

    @Test
    public void processShouldFailFastOnNullArgument(){
        var parser = new CmdLineParser();
        assertThrows(NullPointerException.class, () -> parser.process(null));
    }
    @Test
    public void registerOptionOptionShouldFailFastOnNullArgument(){
        var parser = new CmdLineParser();
        assertThrows(NullPointerException.class, () -> parser.registerOption(null, () -> {}));
    }

    @Test
    public void registerRunnableOptionShouldFailFastOnNullArgument(){
        var parser = new CmdLineParser();
        assertThrows(NullPointerException.class, () -> parser.registerOption("not null", null));
    }

    @Test
    public void registerAlreadyRegisteredOptionShouldFail() {
        var parser = new CmdLineParser();
        parser.registerOption("test", () -> {});
        assertThrows(IllegalStateException.class, () -> parser.registerOption("test", () -> {}));
    }

    @Test
    public void callOneParameterWithoutArguments() throws ParseException {
        String[] arguments = { "-test" };
        AtomicInteger i = new AtomicInteger(0);
        var parser = new CmdLineParser();
        parser.registerOption("test", () -> i.set(1));

        parser.process(arguments);
        assertEquals(1, i.get(), "The 'test' option was not executed !");
    }

    @Test
    public void callTwoParameterWithoutArguments() throws ParseException {
        String[] arguments = { "-test1", "-test2" };
        BitSet bs = new BitSet(2);
        var parser = new CmdLineParser();

        parser.registerOption("test1", () -> bs.set(0));
        parser.registerOption("test2", () -> bs.set(1));

        parser.process(arguments);

        assertTrue(bs.get(0), "The 'test1' option was not executed !");
        assertTrue(bs.get(1), "The 'test2' option was not executed !");
    }

    @Test
    public void callZeroParameterWithoutArgumentsAndWithOneFile() throws ParseException {
        String[] arguments = { "file1", "file2" };
        var parser = new CmdLineParser();

        var files = parser.process(arguments);
        assertArrayEquals(arguments, files.toArray(), "The files passed in arguments was not found");
    }

    @Test
    public void callOneParameterWithOneArgument() throws ParseException {
        String[] arguments = { "-arg1", "663" };
        AtomicInteger i = new AtomicInteger(0);
        var parser = new CmdLineParser();
        parser.registerOption("arg1", 1, args -> i.set(Integer.parseInt(args.get(0))));

        parser.process(arguments);
        assertEquals(663, i.get(), "The 'arg1' option was not executed !");
    }

    @Test
    public void callOneParameterWithOneArgumentAndOneFile() throws ParseException {
        String[] arguments = { "-arg1", "663", "file.txt" };
        AtomicInteger i = new AtomicInteger(0);
        var parser = new CmdLineParser();
        parser.registerOption("arg1", 1, args -> i.set(Integer.parseInt(args.get(0))));

        var files = parser.process(arguments);
        assertEquals(663, i.get(), "The 'arg1' option was not executed !");
        String[] expected = { "file.txt" };
        assertArrayEquals(expected, files.toArray(), "The file passed in argument was not found");
    }

    @Test
    public void callTwoParameterWithOneArgumentButWithoutBeSpecifiedShouldFail() {
        String[] arguments = { "-arg1", "-arg2" };
        var parser = new CmdLineParser();

        parser.registerOption("arg1", 1, args -> {});
        parser.registerOption("arg2", 1, args -> {});

        assertThrows(ParseException.class, () -> parser.process(arguments));
    }

    @Test
    public void callTwoParameterWithOneArgument() throws ParseException {
        String[] arguments = { "-arg1", "4", "-arg2", "7", "file.txt" };
        var parser = new CmdLineParser();
        var bs = new BitSet(8);

        parser.registerOption("arg1", 1, args -> bs.set(Integer.parseInt(args.get(0))) );
        parser.registerOption("arg2", 1, args -> bs.set(Integer.parseInt(args.get(0))) );

        parser.process(arguments);

        assertTrue(bs.get(4), "The 'arg1' option was not executed !");
        assertTrue(bs.get(7), "The 'arg2' option was not executed !");
    }

    @Test
    public void callOneParameterWithTwoArguments() throws ParseException {
        String[] arguments = { "-arg1", "4", "7" };
        var parser = new CmdLineParser();
        var bs = new BitSet(8);

        parser.registerOption("arg1", 2, args -> {
            bs.set(Integer.parseInt(args.get(0)));
            bs.set(Integer.parseInt(args.get(1)));
        } );

        parser.process(arguments);

        assertTrue(bs.get(4) && bs.get(7), "The 'arg1' option was not executed !");
    }

    @Test
    public void callOneParameterWithTwoArgumentsButOnlyOnePrecisedShouldFail() {
        String[] arguments = { "-arg1", "4" };
        var parser = new CmdLineParser();
        var bs = new BitSet(8);

        parser.registerOption("arg1", 2, args -> {
            bs.set(Integer.parseInt(args.get(0)));
            bs.set(Integer.parseInt(args.get(1)));
        } );

        assertThrows(ParseException.class, () -> parser.process(arguments));
    }

}