package fr.uge.poo.cmdline.ex1;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Objects;

public class CmdLineParser {

    private final HashMap<String, Runnable> registeredOptions = new HashMap<>();

    public void registerOption(String option, Runnable runnable) {
        Objects.requireNonNull(option);
        Objects.requireNonNull(runnable);

        var stockRun = registeredOptions.get(option);

        if ( stockRun != null ) {
            throw new IllegalStateException("Argument already defined !");
        }

        registeredOptions.put(option, runnable);
    }

    public List<String> process(String[] arguments) {
        ArrayList<String> files = new ArrayList<>();
        for (String argument : arguments) {
            if (registeredOptions.containsKey(argument)) {
                registeredOptions.get(argument).run();
            } else {
                files.add(argument);
            }
        }
        return files;
    }
}