package fr.uge.poo.cmdline.ex2;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;


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

}