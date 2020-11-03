package fr.uge.poo.cmdline.ex0;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
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

    public static void main(String[] args) {
        var options = new PaintOptions();
        String[] arguments={"-legacy","-no-borders","filename1","filename2"};
        var cmdParser = new CmdLineParser();
        cmdParser.registerOption("-legacy", () -> options.setLegacy(true));
        cmdParser.registerOption("-with-borders", () -> options.setBordered(true));
        cmdParser.registerOption("-no-borders", () -> options.setBordered(false));

        List<String> result = cmdParser.process(arguments);
        List<Path> files = result.stream().map(Path::of).collect(Collectors.toList());

        // this code replaces the rest of the application
        files.forEach(p -> System.out.println(p));
        System.out.println(options.toString());

    }
}
