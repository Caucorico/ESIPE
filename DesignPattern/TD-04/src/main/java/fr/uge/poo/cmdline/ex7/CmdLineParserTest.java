package fr.uge.poo.cmdline.ex7;

import org.junit.jupiter.api.Test;

import java.text.ParseException;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.*;


@SuppressWarnings("static-method")
class CmdLineParserTest {


    /* -------------------------------------------------------------------------------------------------------------- */
    /*                                       Tests on PaintOptionBuilder :                                            */
    /* -------------------------------------------------------------------------------------------------------------- */

    @Test
    public void buildWithNullArgumentShouldFail() {
        assertThrows(NullPointerException.class, () -> new CmdLineParser.PaintOptionBuilder(0, null, null));
    }

    @Test
    public void buildWithNegativeNumberArgumentShouldFail() {
        assertThrows(IllegalArgumentException.class, () -> new CmdLineParser.PaintOptionBuilder(-1, "-test", args -> {}));
    }

    @Test
    public void nullPassedOnOptionalShouldFailInThisCases() {
        var builder = new CmdLineParser.PaintOptionBuilder(0, "-test", args -> {});

        assertAll(
                () -> assertThrows(NullPointerException.class, () -> builder.setDocumentation(null)),
                () -> assertThrows(NullPointerException.class, () -> builder.setAliases(null)),
                () -> assertThrows(NullPointerException.class, () -> builder.addAlias(null)),
                () -> assertThrows(NullPointerException.class, () -> builder.setConflicts(null)),
                () -> assertThrows(NullPointerException.class, () -> builder.addConflict(null))
        );
    }

    @Test
    public void buildWithRequiredParametersShouldWork() {
        Consumer<List<String>> consumer = args -> {};
        var option = new CmdLineParser.PaintOptionBuilder(0, "-test", consumer).build();

        assertAll(
                () -> assertEquals(0, option.getParamNumber()),
                () -> assertEquals("-test", option.getName()),
                () -> assertEquals(consumer, option.getConsumer())
        );
    }

    @Test
    public void buildWithTheRequiredParameterShouldWork() {
        Consumer<List<String>> consumer = args -> {};
        var builder = new CmdLineParser.PaintOptionBuilder(0, "-test", consumer);
        builder.setRequired(true);
        var option = builder.build();
        assertTrue(option.isRequired());
    }

    @Test
    public void setAliasShouldWork() {
        Consumer<List<String>> consumer = args -> {};
        List<String> aliases = new ArrayList<>();
        aliases.add("-alias1");
        aliases.add("-alias2");
        var builder = new CmdLineParser.PaintOptionBuilder(0, "-test", consumer);
        builder.setAliases(aliases);
        var option = builder.build();
        assertArrayEquals(aliases.toArray(), option.getAliases().toArray());
    }

    @Test
    public void addAliasShouldWork() {
        Consumer<List<String>> consumer = args -> {};
        List<String> aliases = new ArrayList<>();
        aliases.add("-alias1");
        aliases.add("-alias2");
        var builder = new CmdLineParser.PaintOptionBuilder(0, "-test", consumer);
        builder.addAlias("-alias1");
        builder.addAlias("-alias2");
        var option = builder.build();
        assertArrayEquals(aliases.toArray(), option.getAliases().toArray());
    }

    @Test
    public void setConflictsShouldWork() {
        Consumer<List<String>> consumer = args -> {};
        List<String> conflicts = new ArrayList<>();
        conflicts.add("-conflict1");
        conflicts.add("-conflict2");
        var builder = new CmdLineParser.PaintOptionBuilder(0, "-test", consumer);
        builder.setConflicts(conflicts);
        var option = builder.build();
        assertArrayEquals(conflicts.toArray(), option.getConflicts().toArray());
    }

    @Test
    public void addConflictShouldWork() {
        Consumer<List<String>> consumer = args -> {};
        List<String> conflicts = new ArrayList<>();
        conflicts.add("-conflict1");
        conflicts.add("-conflict2");
        var builder = new CmdLineParser.PaintOptionBuilder(0, "-test", consumer);
        builder.addConflict("-conflict1");
        builder.addConflict("-conflict2");
        var option = builder.build();
        assertArrayEquals(conflicts.toArray(), option.getConflicts().toArray());
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    /*                                          Tests on CmdLineParser :                                              */
    /* -------------------------------------------------------------------------------------------------------------- */

    @Test
    public void processShouldFailFastOnNullArgument(){
        var parser = new CmdLineParser();
        assertThrows(NullPointerException.class, () -> parser.process(null));
    }
    @Test
    public void registerOptionOptionShouldFailFastOnNullArgument(){
        var parser = new CmdLineParser();
        assertThrows(NullPointerException.class, () -> parser.registerOption(null));
    }

    @Test
    public void registerAlreadyRegisteredOptionShouldFail() {
        var parser = new CmdLineParser();
        var optionBuilder = new CmdLineParser.PaintOptionBuilder(0, "-test", args -> {});
        parser.registerOption(optionBuilder.build());

        var optionBuilder2 = new CmdLineParser.PaintOptionBuilder(0, "-test", args -> {});
        assertThrows(IllegalStateException.class, () -> parser.registerOption(optionBuilder2.build()));
    }

    @Test
    public void callOneParameterWithoutArguments() throws ParseException {
        String[] arguments = { "-test" };
        AtomicInteger i = new AtomicInteger(0);
        var parser = new CmdLineParser();
        var optionBuilder = new CmdLineParser.PaintOptionBuilder(0, "-test", args -> i.set(1));
        parser.registerOption(optionBuilder.build());

        parser.process(arguments);
        assertEquals(1, i.get(), "The 'test' option was not executed !");
    }

    @Test
    public void callTwoParameterWithoutArguments() throws ParseException {
        String[] arguments = { "-test1", "-test2" };
        BitSet bs = new BitSet(2);
        var parser = new CmdLineParser();

        var optionBuilder = new CmdLineParser.PaintOptionBuilder(0, "-test1", args -> bs.set(0));
        parser.registerOption(optionBuilder.build());

        optionBuilder = new CmdLineParser.PaintOptionBuilder(0, "-test2", args -> bs.set(1));
        parser.registerOption(optionBuilder.build() );

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

        var builder = new CmdLineParser.PaintOptionBuilder(1, "-arg1", args -> i.set(Integer.parseInt(args.get(0))));
        parser.registerOption(builder.build() );

        parser.process(arguments);
        assertEquals(663, i.get(), "The 'arg1' option was not executed !");
    }

    @Test
    public void callOneParameterWithOneArgumentAndOneFile() throws ParseException {
        String[] arguments = { "-arg1", "663", "file.txt" };
        AtomicInteger i = new AtomicInteger(0);
        var parser = new CmdLineParser();

        var builder = new CmdLineParser.PaintOptionBuilder(1, "-arg1", args -> i.set(Integer.parseInt(args.get(0))));
        parser.registerOption(builder.build() );

        var files = parser.process(arguments);
        assertEquals(663, i.get(), "The 'arg1' option was not executed !");
        String[] expected = { "file.txt" };
        assertArrayEquals(expected, files.toArray(), "The file passed in argument was not found");
    }

    @Test
    public void callTwoParameterWithOneArgumentButWithoutBeSpecifiedShouldFail() {
        String[] arguments = { "-arg1", "-arg2" };
        var parser = new CmdLineParser();

        var builder = new CmdLineParser.PaintOptionBuilder(1, "-arg1", args -> {});
        parser.registerOption(builder.build() );

        builder = new CmdLineParser.PaintOptionBuilder(1, "-arg2", args -> {});
        parser.registerOption(builder.build() );

        assertThrows(ParseException.class, () -> parser.process(arguments));
    }

    @Test
    public void callTwoParameterWithOneArgument() throws ParseException {
        String[] arguments = { "-arg1", "4", "-arg2", "7", "file.txt" };
        var parser = new CmdLineParser();
        var bs = new BitSet(8);

        var builder = new CmdLineParser.PaintOptionBuilder(1, "-arg1", args -> bs.set(Integer.parseInt(args.get(0))));
        parser.registerOption(builder.build() );

        builder = new CmdLineParser.PaintOptionBuilder(1, "-arg2", args -> bs.set(Integer.parseInt(args.get(0))));
        parser.registerOption(builder.build() );

        parser.process(arguments);

        assertTrue(bs.get(4), "The 'arg1' option was not executed !");
        assertTrue(bs.get(7), "The 'arg2' option was not executed !");
    }

    @Test
    public void callOneParameterWithTwoArguments() throws ParseException {
        String[] arguments = { "-arg1", "4", "7" };
        var parser = new CmdLineParser();
        var bs = new BitSet(8);

        var builder = new CmdLineParser.PaintOptionBuilder(2, "-arg1", args -> {
            bs.set(Integer.parseInt(args.get(0)));
            bs.set(Integer.parseInt(args.get(1)));
        });
        parser.registerOption(builder.build() );

        parser.process(arguments);

        assertTrue(bs.get(4) && bs.get(7), "The 'arg1' option was not executed !");
    }

    @Test
    public void callOneParameterWithTwoArgumentsButOnlyOnePrecisedShouldFail() {
        String[] arguments = { "-arg1", "4" };
        var parser = new CmdLineParser();
        var bs = new BitSet(8);

        var builder = new CmdLineParser.PaintOptionBuilder(2, "-arg1", args -> {
            bs.set(Integer.parseInt(args.get(0)));
            bs.set(Integer.parseInt(args.get(1)));
        });
        parser.registerOption(builder.build() );

        assertThrows(ParseException.class, () -> parser.process(arguments));
    }

    @Test
    public void callInitialAndAliasShouldFail(){
        String[] arguments = { "-arg1", "--argument1" };
        var parser = new CmdLineParser();

        var builder = new CmdLineParser.PaintOptionBuilder(0, "-arg1", args -> {});
        builder.addAlias("--argument1");

        parser.registerOption(builder.build());

        assertThrows(ParseException.class, () -> parser.process(arguments));
    }

    @Test
    public void callWithAliasOnSimpleArgumentShouldWork() throws ParseException {
        String[] arguments = { "--argument1" };
        var parser = new CmdLineParser();
        AtomicInteger i = new AtomicInteger(0);

        var builder = new CmdLineParser.PaintOptionBuilder(0, "-arg1", args -> i.set(1));
        builder.addAlias("--argument1");

        parser.registerOption(builder.build());
        parser.process(arguments);
        assertEquals(1, i.get());
    }

    @Test
    public void processRequiredOption() {
        var cmdParser = new CmdLineParser();
        var option= new CmdLineParser.PaintOptionBuilder(0, "-test", l->{}).setRequired(true).build();
        var option2= new CmdLineParser.PaintOptionBuilder(0, "-test1", l->{}).build();
        cmdParser.registerOption(option);
        cmdParser.registerOption(option2);
        String[] arguments = {"-test1","a","b"};
        assertThrows(ParseException.class,()->{cmdParser.process(arguments);});
    }

    @Test
    public void processConflicts() {
        var cmdParser = new CmdLineParser();
        var option= new CmdLineParser.PaintOptionBuilder(0, "-test", l->{}).addConflict("-test1").build();
        cmdParser.registerOption(option);
        var option2= new CmdLineParser.PaintOptionBuilder(0, "-test1", l->{}).build();
        cmdParser.registerOption(option2);
        String[] arguments = {"-test","-test1"};
        assertThrows(ParseException.class,()->{cmdParser.process(arguments);});
    }

    @Test
    public void processConflicts2() {
        var cmdParser = new CmdLineParser();
        var option= new CmdLineParser.PaintOptionBuilder(0, "-test", l->{}).addConflict("-test1").build();
        cmdParser.registerOption(option);
        var option2= new CmdLineParser.PaintOptionBuilder(0, "-test1", l->{}).build();
        cmdParser.registerOption(option2);
        String[] arguments = {"-test1","-test"};
        assertThrows(ParseException.class,()->{cmdParser.process(arguments);});
    }

    @Test
    public void processConflictsAndAliases() {
        var cmdParser = new CmdLineParser();
        var option= new CmdLineParser.PaintOptionBuilder(0, "-test", l->{}).addConflict("-test2").build();
        cmdParser.registerOption(option);
        var option2= new CmdLineParser.PaintOptionBuilder(0, "-test1", l->{}).addAlias("-test2").build();
        cmdParser.registerOption(option2);
        String[] arguments = {"-test1","-test"};
        assertThrows(ParseException.class,()->{cmdParser.process(arguments);});
    }

    @Test
    public void processConflictsAndAliases2() {
        var cmdParser = new CmdLineParser();
        var option= new CmdLineParser.PaintOptionBuilder(0, "-test", l->{}).addConflict("-test2").build();
        cmdParser.registerOption(option);
        var option2= new CmdLineParser.PaintOptionBuilder(0, "-test1", l->{}).addAlias("-test2").build();
        cmdParser.registerOption(option2);
        String[] arguments = {"-test","-test1"};
        assertThrows(ParseException.class,()->{cmdParser.process(arguments);});
    }

    @Test
    public void processPolicyStandard() {
        var hosts = new ArrayList<String>();
        var cmdParser = new CmdLineParser();
        var optionHosts= new CmdLineParser.PaintOptionBuilder(2,"-hosts", hosts::addAll).build();
        cmdParser.registerOption(optionHosts);
        var optionLegacy= new CmdLineParser.PaintOptionBuilder(0,"-legacy", args->{}).build();
        cmdParser.registerOption(optionLegacy);
        String[] arguments = {"-hosts","localhost","-legacy","file"};
        assertThrows(ParseException.class,()->{cmdParser.process(arguments,CmdLineParser.STANDARD);});
    }

    @Test
    public void processPolicyRelaxed() throws ParseException {
        var hosts = new ArrayList<String>();
        var cmdParser = new CmdLineParser();
        var optionHosts= new CmdLineParser.PaintOptionBuilder(2,"-hosts", hosts::addAll).build();
        cmdParser.registerOption(optionHosts);
        var optionLegacy= new CmdLineParser.PaintOptionBuilder(0,"-legacy", args->{}).build();
        cmdParser.registerOption(optionLegacy);
        String[] arguments = {"-hosts","localhost","-legacy","file"};
        cmdParser.process(arguments,CmdLineParser.RELAXED);
        assertEquals(1,hosts.size());
        assertEquals("localhost",hosts.get(0));
    }



    @Test
    public void processPolicyOldSchool() throws ParseException {
        var hosts = new ArrayList<String>();
        var cmdParser = new CmdLineParser();
        var optionHosts= new CmdLineParser.PaintOptionBuilder(2,"-hosts", hosts::addAll).build();
        cmdParser.registerOption(optionHosts);
        var optionLegacy= new CmdLineParser.PaintOptionBuilder(0,"-legacy", args->{}).build();
        cmdParser.registerOption(optionLegacy);
        String[] arguments = {"-hosts","localhost","-legacy","file"};
        cmdParser.process(arguments,CmdLineParser.OLDSCHOOL);
        assertEquals(2,hosts.size());
        assertEquals("localhost",hosts.get(0));
        assertEquals("-legacy",hosts.get(1));
    }

}