package fr.uge.poo.visitors.stp;

import com.evilcorp.stp.*;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Optional;
import java.util.Scanner;

public class Triviale {

    public static void main(String[] args) {
        var scan = new Scanner(System.in);
        while(scan.hasNextLine()){
            var line = scan.nextLine();
            if (line.equals("quit")){
                break;
            }
            Optional<STPCommand> answer = STPParser.parse(line);
            if (!answer.isPresent()){
                System.out.println("Pas compris");
                continue;
            }
            STPCommand cmd = answer.get();
            if (cmd instanceof HelloCmd){
                System.out.println("Au revoir");
            } else {
                System.out.println("Non implémenté");
            }
        }
    }

}
