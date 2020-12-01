package fr.uge.poo.visitors.stp;

import com.evilcorp.stp.*;

import java.util.Optional;
import java.util.Scanner;

public class Triviale2 {

    public static class Visitored implements VisitorCmd {

        @Override
        public void visit(ElapsedTimeCmd cmd) {
            System.out.println("Non implémenté");
        }

        @Override
        public void visit(HelloCmd cmd) {
            System.out.println("Au revoir");
        }

        @Override
        public void visit(StartTimerCmd cmd) {
            System.out.println("Non implémenté");
        }

        @Override
        public void visit(StopTimerCmd cmd) {
            System.out.println("Non implémenté");
        }
    }

    public static void main(String[] args) {
        var scan = new Scanner(System.in);
        var visitored = new Visitored();
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
            answer.get().accept(visitored);
        }
    }

}
