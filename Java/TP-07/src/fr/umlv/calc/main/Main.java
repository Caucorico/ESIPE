package fr.umlv.calc.main;

import fr.umlv.calc.Expr;
import fr.umlv.calc.OpOrValue;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main
{
    public static void main(String[] args) {
        Expr expression;

        List<String> list = new ArrayList<>();
        Scanner scanner = new Scanner(System.in);



        expression = Expr.parse(scanner);
        Expr.display(expression);
    }
}
