package fr.uge.poo.cmdline.ex6;

import java.net.InetSocketAddress;
import java.text.ParseException;
import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.IntConsumer;

public class CmdLineParser {

    private final OptionsManager optionsManager = new OptionsManager();
    private final DocumentationObserver documentationObserver = new DocumentationObserver();

    public static class PaintOption {
        private final boolean required;
        private final int paramNumber;
        private final String name;
        private final Consumer<List<String>> consumer;
        private final List<String> aliases;
        private final List<String> conflicts;
        private final String documentation;
        private boolean used;

        PaintOption(PaintOptionBuilder builder) {
            this.paramNumber = builder.paramNumber;
            this.name = builder.name;
            this.consumer = builder.consumer;
            this.required = builder.required;
            this.aliases = builder.aliases;
            this.conflicts = builder.conflicts;
            this.documentation = builder.documentation;
        }

        public boolean isRequired() {
            return required;
        }

        public int getParamNumber() {
            return paramNumber;
        }

        public String getName() {
            return name;
        }

        public Consumer<List<String>> getConsumer() {
            return consumer;
        }

        public List<String> getAliases() {
            return aliases;
        }

        public List<String> getConflicts() {
            return conflicts;
        }

        public String getDocumentation() {
            return documentation;
        }

        public boolean isUsed() {
            return used;
        }
    }

    public static class PaintOptionBuilder {

        private int paramNumber;
        private String name;
        private Consumer<List<String>> consumer;
        private boolean required;
        private List<String> aliases = new ArrayList<>(1);
        private List<String> conflicts = new ArrayList<>(1);
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
            this.documentation = Objects.requireNonNull(documentation);
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

        public PaintOptionBuilder setConflicts(List<String> conflicts) {
            Objects.requireNonNull(conflicts);
            this.conflicts = conflicts;
            return this;
        }

        public PaintOptionBuilder addConflict(String conflict) {
            Objects.requireNonNull(conflict);
            this.conflicts.add(conflict);
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

    interface OptionsObserver {

        void onRegisteredOption(OptionsManager optionsManager, PaintOption option);

        void onProcessedOption(OptionsManager optionsManager, PaintOption option);

        void onFinishedProcess(OptionsManager optionsManager) throws ParseException;
    }

    static class LoggerObserver implements OptionsObserver {
        @Override
        public void onRegisteredOption(OptionsManager optionsManager, PaintOption option) {
            System.out.println("Option "+option+ " is registered");
        }

        @Override
        public void onProcessedOption(OptionsManager optionsManager, PaintOption option) {
            System.out.println("Option "+option+ " is processed");
        }

        @Override
        public void onFinishedProcess(OptionsManager optionsManager) {
            System.out.println("Process method is finished");
        }
    }

    private static class RequireObserver implements OptionsObserver {
        @Override
        public void onRegisteredOption(OptionsManager optionsManager, PaintOption option) {
        }

        @Override
        public void onProcessedOption(OptionsManager optionsManager, PaintOption option) {
            if ( option == null ) return;
            option.used = true;
        }

        @Override
        public void onFinishedProcess(OptionsManager optionsManager) throws ParseException {
            for ( var option : optionsManager.byName.values()) {
                if ( option.required && !option.used ) {
                    throw new ParseException("Option " + option.name + " ", 0);
                }
            }
        }
    }

    static class DocumentationObserver implements OptionsObserver {
        List<PaintOption> options = new ArrayList<>(1);

        @Override
        public void onRegisteredOption(OptionsManager optionsManager, PaintOption option) {
            options.add(option);
        }

        @Override
        public void onProcessedOption(OptionsManager optionsManager, PaintOption option) {
        }

        @Override
        public void onFinishedProcess(OptionsManager optionsManager) {
        }

        public String usage() {
            StringBuilder builder = new StringBuilder();
            for ( var option : options ) {
                builder.append(option.name).append(" : ").append(option.documentation).append('\n');
            }

            return builder.toString();
        }
    }

    private static class ConflictObserver implements OptionsObserver {

        private final List<PaintOption> optionList = new ArrayList<>(1);

        @Override
        public void onRegisteredOption(OptionsManager optionsManager, PaintOption option) {
        }

        @Override
        public void onProcessedOption(OptionsManager optionsManager, PaintOption option) {
            if( option == null ) return;
            optionList.add(option);
        }

        @Override
        public void onFinishedProcess(OptionsManager optionsManager) throws ParseException {
            Objects.requireNonNull(optionsManager);

            for ( int i = 0 ; i < optionList.size() ; i++ ) {
                var option = optionList.get(i);

                for ( int j = i+1 ; j < optionList.size() ; j++ ) {
                    var check = optionList.get(j);

                    /* Check if we don't use the same argument twice */
                    if ( option.name.equals(check.name) ) {
                        throw new ParseException("Argument " + option.name + " in conflict with " + check.name + ".", 0);
                    }

                    /* Check if we don't use the same argument twice with its aliases */
                    for ( var alias : option.aliases ) {
                        if ( check.name.equals(alias) ) {
                            throw new ParseException("Argument " + check.name + " in conflict with " + alias + ".", 0);
                        }
                    }

                    /* Check if the argument isn't in conflict with the current argument : */
                    for ( var conflict : option.conflicts ) {
                        if ( check.name.equals(conflict) ) {
                            throw new ParseException("Argument " + check.name + " in conflict with " + conflict + ".", 0);
                        }

                        for ( var alias : check.aliases ) {
                            if ( alias.equals(conflict) ) {
                                throw new ParseException("Argument " + alias + " in conflict with " + conflict + ".", 0);
                            }
                        }
                    }

                    /* Check if the current argument isn't in conflict with the argument : */
                    for ( var conflict : check.conflicts ) {
                        if ( option.name.equals(conflict) ) {
                            throw new ParseException("Argument " + option.name + " in conflict with " + conflict + ".", 0);
                        }

                        for ( var alias : option.aliases ) {
                            if ( alias.equals(conflict) ) {
                                throw new ParseException("Argument " + alias + " in conflict with " + conflict + ".", 0);
                            }
                        }
                    }
                }
            }

            optionList.clear();
        }
    }

    private static class OptionsManager {

        private final HashMap<String, PaintOption> byName = new HashMap<>();

        private final List<OptionsObserver> observers = new ArrayList<>();

        /**
         * Register the option with all its possible names
         * @param option
         */
        void register(PaintOption option) {
            Objects.requireNonNull(option);

            /* Notify all the observers */
            for ( var observer : observers ) {
                observer.onRegisteredOption(this, option);
            }

            register(option.name, option);
            for (var alias : option.aliases) {
                register(alias, option);
            }
        }

        private void register(String name, PaintOption option) {
            Objects.requireNonNull(name);
            Objects.requireNonNull(option);

            /* Notify all the observers */
            for ( var observer : observers ) {
                observer.onRegisteredOption(this, option);
            }

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

        Optional<PaintOption> processOption(String optionName) throws ParseException {
            Objects.requireNonNull(optionName);

            var option = this.byName.get(optionName);
            var optional = Optional.ofNullable(option);

            /* Notify all the observers */
            for ( var observer : observers ) {
                if ( optional.isPresent() ) {
                    observer.onProcessedOption(this, optional.get());
                } else {
                    observer.onProcessedOption(this, null);
                }
            }

            return optional;
        }

        /**
         * This method is called to signal the method process of the CmdLineParser is finished
         */
        void finishProcess() throws ParseException {
            /* Notify all the observers */
            for ( var observer : observers ) {
                observer.onFinishedProcess(this);
            }
        }

        void registerOptionsObserver(OptionsObserver observer) {
            Objects.requireNonNull(observer);
            observers.add(observer);
        }
    }

    public CmdLineParser() {
        CmdLineParser.OptionsObserver observer = new CmdLineParser.RequireObserver();
        CmdLineParser.OptionsObserver observer2 = new CmdLineParser.ConflictObserver();
        optionsManager.observers.add(observer);
        optionsManager.observers.add(observer2);
        optionsManager.observers.add(documentationObserver);
    }

    public void registerOption(PaintOption paintOption) {
        Objects.requireNonNull(paintOption);
        optionsManager.register(paintOption);
    }

    public void registerObserver( OptionsObserver observer ) {
        optionsManager.registerOptionsObserver(observer);
    }

    public List<String> process(String[] arguments) throws ParseException {
        ArrayList<String> files = new ArrayList<>();
        int i = 0;

        var t = Integer.class;

        while ( i < arguments.length ) {
            var argument = arguments[i];
            var existing = this.optionsManager.processOption(argument);
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

                existing.get().consumer.accept(params);
                i += ( option.paramNumber + 1);
            } else {
                files.add(argument);
                i++;
            }
        }

        /* In the finally ?*/
        optionsManager.finishProcess();

        return files;
    }

    public void usage() {
        System.out.println(documentationObserver.usage());
    }

}