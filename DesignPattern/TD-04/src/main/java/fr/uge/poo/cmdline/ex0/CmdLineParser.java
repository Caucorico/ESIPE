package fr.uge.poo.cmdline.ex0;

import java.util.*;

public class CmdLineParser {

    private final HashSet<String> registeredOptions = new HashSet<>();
    private final HashSet<String> optionsSeen = new HashSet<>();

    public void registerOption(String option) {
        Objects.requireNonNull(option);
        registeredOptions.add(option);
    }

    public Set<String> getOptionsSeen() {
        return Collections.unmodifiableSet(optionsSeen);
    }

    public List<String> process(String[] arguments) {
        ArrayList<String> files = new ArrayList<>();
        for (String argument : arguments) {
            if (registeredOptions.contains(argument)) {
                optionsSeen.add(argument);
            } else {
                files.add(argument);
            }
        }
        return files;
    }
}