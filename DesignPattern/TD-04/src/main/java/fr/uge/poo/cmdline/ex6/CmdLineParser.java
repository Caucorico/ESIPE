package fr.uge.poo.cmdline.ex6;

import java.text.ParseException;
import java.util.*;
import java.util.function.Consumer;

public class CmdLineParser {

    private final HashMap<String, PaintOption> registeredOptions = new HashMap<>();

    private static class PaintOption {

        final int paramNumber;
        final String name;
        final Consumer<List<String>> consumer;
        final List<String> aliases;

        PaintOption(int paramNumber, String name, Consumer<List<String>> consumer, List<String> aliases) {
            this.paramNumber = paramNumber;
            this.name = name;
            this.consumer = consumer;
            this.aliases = aliases;
        }

    }


    private static class OptionsManager {

        private final HashMap<String, PaintOption> byName = new HashMap<>();

        /**
         * Register the option with all its possible names
         * @param option
         */
        void register(PaintOption option) {
            register(option.name, option);
            for (var alias : option.aliases) {
                register(alias, option);
            }
        }

        private void register(String name, PaintOption option) {
            if (byName.containsKey(name)) {
                throw new IllegalStateException("Option " + name + " is already registered.");
            }
            byName.put(name, option);
        }

        /**
         * This method is called to signal that an option is encountered during
         * a command line process
         *
         * @param optionName
         * @return the corresponding object option if it exists
         */

        Optional<PaintOption> processOption(String optionName) {
            return Optional.ofNullable(byName.get(optionName));
        }

        /**
         * This method is called to signal the method process of the CmdLineParser is finished
         */
        void finishProcess() {

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
        option = "-" + option;

        var stockRun = registeredOptions.get(option);

        if ( stockRun != null ) {
            throw new IllegalStateException("Argument already defined !");
        }

//        registeredOptions.put(option, new PaintOption(argumentNumber, option, consumer));
    }

    public List<String> process(String[] arguments) throws ParseException {
        ArrayList<String> files = new ArrayList<>();
        int i = 0;

        var t = Integer.class;

        while ( i < arguments.length ) {
            var argument = arguments[i];
            var existing = registeredOptions.get(argument);
            if ( existing != null ) {
                if ( arguments.length - i <= existing.paramNumber ) {
                    throw new ParseException("Not enough argument !", existing.paramNumber - argument.length() - i);
                }
                ArrayList<String> params = new ArrayList<>(existing.paramNumber);
                params.addAll(Arrays.asList(arguments).subList(i+1, existing.paramNumber + i+1));

                for ( var k = 0 ; k < params.size() ; k++ ) {
                    var param = params.get(k);
                    if ( param.startsWith("-") ) {
                        throw new ParseException("Not enough argument, the param " + param + " is encounter before the end of the arguments number", i+k);
                    }
                }

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