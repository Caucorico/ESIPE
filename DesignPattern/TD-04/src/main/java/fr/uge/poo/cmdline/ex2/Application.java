package fr.uge.poo.cmdline.ex2;

import java.nio.file.Path;
import java.text.ParseException;
import java.util.List;
import java.util.stream.Collectors;

public class Application {

    static private class PaintOptions{
        private boolean legacy=false;
        private boolean bordered=true;

        public void setLegacy(boolean legacy) {
            this.legacy = legacy;
        }

        public boolean isLegacy(){
            return legacy;
        }

        public void setBordered(boolean bordered){
            this.bordered=bordered;
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
        var options = new PaintOptions();
        String[] arguments={"-legacy","-border-width", "65", "-min-size", "12", "12", "filename1","filename2"};
        var cmdParser = new CmdLineParser();
        cmdParser.registerOption("-legacy", () -> options.setLegacy(true));
        cmdParser.registerOption("-with-borders", () -> options.setBordered(true));
        cmdParser.registerOption("-no-borders", () -> options.setBordered(false));
        cmdParser.registerOption("-border-width", 1, argss -> {
            options.setBordered(true);
            System.out.println("coucou " + argss.get(0));
        });
        cmdParser.registerOption("-min-size", 2, argss -> {
            options.setBordered(true);
            System.out.println("coucou2 " + argss.get(0) + " " + argss.get(1));
        });

        List<String> result = cmdParser.process(arguments);
        List<Path> files = result.stream().map(Path::of).collect(Collectors.toList());

        // this code replaces the rest of the application
        files.forEach(System.out::println);
        System.out.println(options.toString());

    }
}
