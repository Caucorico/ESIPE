package fr.uge.poo.cmdline.ex5;

import java.net.InetSocketAddress;
import java.text.ParseException;
import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.IntConsumer;

public class CmdLineParser {

    private final HashMap<String, PaintOption> registeredOptions = new HashMap<>();

    private static class PaintOption {
        final boolean required;
        final int paramNumber;
        final String name;
        final Consumer<List<String>> consumer;
        final List<String> aliases;
        final String documentation;
        private boolean used;

        PaintOption(PaintOptionBuilder builder) {
            this.paramNumber = builder.paramNumber;
            this.name = builder.name;
            this.consumer = builder.consumer;
            this.required = builder.required;
            this.aliases = builder.aliases;
            this.documentation = builder.documentation;
        }

    }

    public static class PaintOptionBuilder {

        private int paramNumber;
        private String name;
        private Consumer<List<String>> consumer;
        private boolean required;
        private List<String> aliases = new ArrayList<>(1);
        private String documentation;

        public PaintOptionBuilder(int paramNumber, String name, Consumer<List<String>> consumer) {
            if ( paramNumber < 0 ) {
                throw new IllegalArgumentException();
            }
            this.paramNumber = paramNumber;
            this.name = Objects.requireNonNull(name);
            this.consumer = Objects.requireNonNull(consumer);
        }

        public PaintOptionBuilder setRequired(boolean required) {
            this.required = required;
            return this;
        }

        public PaintOptionBuilder setDocumentation(String documentation) {
            this.documentation = documentation;
            return this;
        }

        public PaintOptionBuilder setAliases(List<String> aliases) {
            this.aliases = Objects.requireNonNull(aliases);
            return this;
        }

        public PaintOptionBuilder addAlias(String alias) {
            Objects.requireNonNull(alias);
            this.aliases.add(alias);
            return this;
        }

        public PaintOption build() {
            return new PaintOption(this);
        }

        public static PaintOptionBuilder simpleIntegerBuilder(String name, IntConsumer intConsumer) {
            return new PaintOptionBuilder(
                    1,
                    name,
                    args -> intConsumer.accept( Integer.parseInt(args.get(0)) )
            );
        }

        public static PaintOptionBuilder doubleIntegerBuilder(String name, BiConsumer<Integer, Integer> intConsumer) {
            return new PaintOptionBuilder(
                    2,
                    name,
                    args -> intConsumer.accept( Integer.parseInt(args.get(0)), Integer.parseInt(args.get(1)) )
            );
        }

        public static PaintOptionBuilder simpleInetSocketAddressBuilder(String name, Consumer<InetSocketAddress> consumer) {
            return new PaintOptionBuilder(
                    2,
                    name,
                    args -> consumer.accept( new InetSocketAddress(args.get(0), Integer.parseInt(args.get(1))) )
            );
        }
    }

    public static class IntegerPaintOptionBuilder {
        
    }

    public void registerOption(PaintOption paintOption) {
        Objects.requireNonNull(paintOption);

        var stockOpt = registeredOptions.get(paintOption.name);

        if ( stockOpt != null ) {
            throw new IllegalStateException("Argument " + paintOption.name + " already defined !");
        }

        registeredOptions.put(paintOption.name, paintOption);

        for ( var alias : paintOption.aliases ) {
            stockOpt = registeredOptions.get(alias);
            if ( stockOpt != null ) {
                throw new IllegalStateException("Argument " + paintOption.name + " already defined !");
            }
            registeredOptions.put(alias, paintOption);
        }

    }

    public Optional<PaintOption> getOption(String optionName) throws ParseException {
        Objects.requireNonNull(optionName);
        var option = this.registeredOptions.get(optionName);
        var optional = Optional.ofNullable(option);
        if ( optional.isPresent() && option.used ) {
            throw new ParseException("Argument already used", 0);
        }

        optional.ifPresent( o -> o.used = true);
        return optional;
    }

    public List<String> process(String[] arguments) throws ParseException {
        ArrayList<String> files = new ArrayList<>();
        int i = 0;

        var t = Integer.class;

        while ( i < arguments.length ) {
            var argument = arguments[i];
            var existing = this.getOption(argument);
            if ( existing.isPresent() ) {
                var option = existing.get();
                if ( arguments.length - i <= option.paramNumber ) {
                    throw new ParseException("Not enough argument !", option.paramNumber - argument.length() - i);
                }
                ArrayList<String> params = new ArrayList<>(option.paramNumber);
                params.addAll(Arrays.asList(arguments).subList(i+1, option.paramNumber + i+1));

                for ( var k = 0 ; k < params.size() ; k++ ) {
                    var param = params.get(k);
                    if ( param.startsWith("-") ) {
                        throw new ParseException("Not enough argument, the param " + param + " is encounter before the end of the arguments number", i+k);
                    }
                }

                registeredOptions.get(argument).consumer.accept(params);
                i += ( option.paramNumber + 1);
            } else {
                files.add(argument);
                i++;
            }
        }
        return files;
    }
}