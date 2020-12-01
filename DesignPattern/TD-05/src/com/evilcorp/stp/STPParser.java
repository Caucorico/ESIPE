package com.evilcorp.stp;

import java.util.Arrays;
import java.util.Objects;
import java.util.Optional;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class STPParser {

	static Pattern NUMBER = Pattern.compile("[0-9]+");
	static Pattern SPACES = Pattern.compile(" +");

	/**
	 * Test if the string s represents a number
	 * @param s    the string to be checked
	 */
	private static boolean isInteger(String s){
		return NUMBER.matcher(s).matches();
	}

	/**
	 * Convert a text command to the corresponding object in the the {@link STPCommand}.
	 * If the format of cmd is invalid, returns Optional.empty()
	 * @param cmd
	 *         the line to be parsed. Must be non-null.
	 * @return the corresponding {@link STPCommand} wrapped in a {@link Optional}
	 */
	public static Optional<STPCommand> parse(String cmd){
		Objects.requireNonNull(cmd);
		var tokens = SPACES.split(cmd);
		if (tokens.length==0){
			return Optional.empty();
		}
		switch (tokens[0]){
		case "hello": 
			return Optional.of(new HelloCmd());
		case "start": 
			if (tokens.length!=2 || !isInteger(tokens[1])){
				return Optional.empty();
			}
			return Optional.of(new StartTimerCmd(Integer.parseInt(tokens[1])));
		case "stop": 
			if (tokens.length!=2 || !isInteger(tokens[1])){
				return Optional.empty();
			}
			return Optional.of(new StopTimerCmd(Integer.parseInt(tokens[1])));
		case "elapsed": 
			if (tokens.length==1){
				return Optional.empty();
			}
			for(int i=1;i<tokens.length;i++){
				if (!isInteger(tokens[i])){
					return Optional.empty();
				}
			}
			var timersId = Arrays.stream(tokens).skip(1).map(Integer::parseInt).collect(Collectors.toUnmodifiableList());
			return Optional.of(new ElapsedTimeCmd(timersId));
		default:
			return Optional.empty();
		}
	}
}

