package exercice01;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class Main {

    private void stopThreadById(int id) {
        
    }

    private void readStdIn() throws IOException {
        System.out.println("enter a thread id:");
        try (var input = new InputStreamReader(System.in);
             var reader = new BufferedReader(input)) {
            String line;
            while ((line = reader.readLine()) != null) {
                var threadId = Integer.parseInt(line);

            }
        }
    }

    public static void main(String[] args) {

    }
}
