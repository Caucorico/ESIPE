package fr.uge.poo.cmdline.ex1;

import static org.junit.jupiter.api.Assertions.*;

import fr.uge.poo.cmdline.ex1.CmdLineParser;
import org.junit.jupiter.api.Test;


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