package fr.uge.poo.cmdline.ex3;

import java.text.ParseException;
import java.util.*;
import java.util.function.Consumer;

public class CmdLineParser {

    private final HashMap<String, PaintOption> registeredOptions = new HashMap<>();

    private static class PaintOption {

        final int paramNumber;
        final String name;
        final Consumer<List<String>> consumer;

        PaintOption(int paramNumber, String name, Consumer<List<String>> consumer) {
            this.paramNumber = paramNumber;
            this.name = name;
            this.consumer = consumer;
        }

    }

    public void registerOption(String option, Runnable runnable) {
        Objects.requireNonNull(option);
        Objects.requireNonNull(runnable);

        registerOption(option, 0, __ -> runnable.run());
    }

    public void registerOption(String option, int argumentNumber, Consumer<List<String>> consumer) {
        Objects.requireNonNull(option);
        Objects.requireNonNull(consumer);

        var stockRun = registeredOptions.get(option);

        if ( stockRun != null ) {
            throw new IllegalStateException("Argument already defined !");
        }

        registeredOptions.put(option, new PaintOption(argumentNumber, option, consumer));
    }

    public List<String> process(String[] arguments) throws ParseException {
        ArrayList<String> files = new ArrayList<>();
        int i = 0;

        while ( i < arguments.length ) {
            var argument = arguments[i];
            var existing = registeredOptions.get(argument);
            if ( existing != null ) {
                if ( argument.length() - i < existing.paramNumber ) {
                    throw new ParseException("Not enough argument !", existing.paramNumber - argument.length() - i);
                }
                ArrayList<String> params = new ArrayList<>(existing.paramNumber);
                params.addAll(Arrays.asList(arguments).subList(i+1, existing.paramNumber + i+1));
                registeredOptions.get(argument).consumer.accept(params);
                i += ( existing.paramNumber + 1);
            } else {
                files.add(argument);
                i++;
            }
        }
        return files;
    }
}