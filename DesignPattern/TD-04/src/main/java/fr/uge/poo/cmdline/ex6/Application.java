package fr.uge.poo.cmdline.ex6;

import java.nio.file.Path;
import java.text.ParseException;
import java.util.List;
import java.util.stream.Collectors;

public class Application {

    static private class PaintOptionsBuilder {

        private boolean legacy = false;
        private boolean bordered = true;

        public PaintOptionsBuilder setLegacy(boolean legacy) {
            this.legacy = legacy;
            return this;
        }

        public PaintOptionsBuilder setBordered(boolean bordered) {
            this.bordered = bordered;
            return this;
        }

        public PaintOptions build() {
            return new PaintOptions(this);
        }
    }

    static private class PaintOptions{
        private final boolean legacy;
        private final boolean bordered;

        private PaintOptions(PaintOptionsBuilder builder) {
            this.legacy = builder.legacy;
            this.bordered = builder.bordered;
        }

        public boolean isLegacy(){
            return legacy;
        }

        public boolean isBordered(){
            return bordered;
        }

        @Override
        public String toString(){
            return "PaintOption [ bordered = "+bordered+", legacy = "+ legacy +" ]";
        }
    }

    public static void main(String[] args) throws ParseException {
        var optionsBuilder = new PaintOptionsBuilder();

        String[] arguments={"-legacy","-border-width", "65", "-min-size", "12", "12", "filename1","filename2"};
        var cmdParser = new CmdLineParser();

        var optionBuilder = new CmdLineParser.PaintOptionBuilder(0, "-legacy", argss -> optionsBuilder.setLegacy(true));
        cmdParser.registerOption(optionBuilder.build());

        optionBuilder = new CmdLineParser.PaintOptionBuilder(0, "-with-borders", argss -> optionsBuilder.setBordered(true));
        cmdParser.registerOption(optionBuilder.build());

        optionBuilder = new CmdLineParser.PaintOptionBuilder(0, "-no-borders", argss -> optionsBuilder.setBordered(false));
        cmdParser.registerOption(optionBuilder.build());

        optionBuilder = new CmdLineParser.PaintOptionBuilder(1, "-border-width", argss -> {
            optionsBuilder.setBordered(true);
            System.out.println("coucou " + argss.get(0));
        });
        cmdParser.registerOption(optionBuilder.build());

        optionBuilder = new CmdLineParser.PaintOptionBuilder(2, "-min-size", argss -> {
            optionsBuilder.setBordered(true);
            System.out.println("coucou2 " + argss.get(0) + " " + argss.get(1));
        });
        cmdParser.registerOption(optionBuilder.build());

        List<String> result = cmdParser.process(arguments);
        List<Path> files = result.stream().map(Path::of).collect(Collectors.toList());

        var options = optionsBuilder.build();

        // this code replaces the rest of the application
        files.forEach(System.out::println);
        System.out.println(options.toString());

    }
}
