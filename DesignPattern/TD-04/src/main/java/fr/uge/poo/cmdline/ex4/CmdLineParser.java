package fr.uge.poo.cmdline.ex4;

import java.text.ParseException;
import java.util.*;
import java.util.function.Consumer;

public class CmdLineParser {

    private final HashMap<String, PaintOption> registeredOptions = new HashMap<>();

    private static class PaintOption {
        final boolean required;
        final int paramNumber;
        final String name;
        final Consumer<List<String>> consumer;

        PaintOption(PaintOptionBuilder builder) {
            this.paramNumber = builder.paramNumber;
            this.name = builder.name;
            this.consumer = builder.consumer;
            this.required = builder.required;
        }

    }

    public static class PaintOptionBuilder {

        private int paramNumber;
        private String name;
        private Consumer<List<String>> consumer;
        private boolean required;
        /*private List<String> aliases;*/

        public PaintOptionBuilder setParamNumber(int paramNumber) {
            if ( paramNumber < 0 ) {
                throw new IllegalArgumentException();
            }
            this.paramNumber = paramNumber;
            return this;
        }

        public PaintOptionBuilder setName(String name) {
            this.name = Objects.requireNonNull(name);
            return this;
        }

        public PaintOptionBuilder setConsumer(Consumer<List<String>> consumer) {
            this.consumer = Objects.requireNonNull(consumer);
            return this;
        }

        public void setRequired(boolean required) {
            this.required = required;
        }

        /*public PaintOptionBuilder setAliases(List<String> aliases) {
            this.aliases = Objects.requireNonNull(aliases);
            return this;
        }

        public PaintOptionBuilder addAlias(String alias) {
            Objects.requireNonNull(alias);
            this.aliases.add(alias);
            return this;
        }*/

        public PaintOption build() {
            /* Todo : check if all requirements are ok */
            return new PaintOption(this);
        }
    }

    public void registerOption(PaintOption paintOption) {
        Objects.requireNonNull(paintOption);

        var stockOpt = registeredOptions.get(paintOption.name);

        if ( stockOpt != null ) {
            throw new IllegalStateException("Argument already defined !");
        }

        registeredOptions.put(paintOption.name, paintOption);

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