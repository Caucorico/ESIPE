package fr.uge.poo.visitors.stp;
import com.evilcorp.stp.*;

import java.time.LocalDateTime;
import java.util.*;

public class Application {

    public static class Visitored implements VisitorCmd {

        public interface Observer {
            void onElapsedTime();
            void onStartTimer();
            void onHello();
            void onStopTimer();
        }

        public static class NumberTimeCalledObserver implements Observer {

            private int elapsedTimeCounter;
            private int startTimerCounter;
            private int helloCounter;
            private int stopTimerCounter;

            @Override
            public void onElapsedTime() {
                elapsedTimeCounter++;
            }

            @Override
            public void onStartTimer() {
                startTimerCounter++;
            }

            @Override
            public void onHello() {
                helloCounter++;
            }

            @Override
            public void onStopTimer() {
                stopTimerCounter++;
            }
        }



        HashMap<Integer,Long> timers = new HashMap<>();

        @Override
        public void visit(ElapsedTimeCmd cmd) {
            ElapsedTimeCmd ellapsedTimersCmd = (ElapsedTimeCmd) cmd;
            var currentTime =  System.currentTimeMillis();
            for(var timerId : ellapsedTimersCmd.getTimers()){
                var startTime = timers.get(timerId);
                if (startTime==null){
                    System.out.println("Unknown timer "+timerId);
                    continue;
                }
                System.out.println("Ellapsed time on timerId "+timerId+" : "+(currentTime-startTime)+"ms");
            }
        }

        @Override
        public void visit(HelloCmd cmd) {
            System.out.println("Hello the current date is "+ LocalDateTime.now());
        }

        @Override
        public void visit(StartTimerCmd cmd) {
            StartTimerCmd startTimerCmd = (StartTimerCmd) cmd;
            var timerId = startTimerCmd.getTimerId();
            if (timers.get(timerId)!=null){
                System.out.println("Timer "+timerId+" was already started");
                return;
            }
            var currentTime =  System.currentTimeMillis();
            timers.put(timerId,currentTime);
            System.out.println("Timer "+timerId+" started");
        }

        @Override
        public void visit(StopTimerCmd cmd) {
            StopTimerCmd stopTimerCmd = (StopTimerCmd) cmd;
            var timerId = stopTimerCmd.getTimerId();
            var startTime = timers.get(timerId);
            if (startTime==null){
                System.out.println("Timer "+timerId+" was never started");
                return;
            }
            var currentTime =  System.currentTimeMillis();
            System.out.println("Timer "+timerId+" was stopped after running for "+(currentTime-startTime)+"ms");
            timers.put(timerId,null);
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
                System.out.println("Unrecognized command");
                continue;
            }
            answer.get().accept(visitored);
        }
    }
}
