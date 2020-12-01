package fr.uge.poo.visitors.expr;

import java.util.Iterator;
import java.util.regex.Pattern;

public class Main {

    public static void main(String[] args) {
        ExprVisitor<Integer> v = new EvalExprVisitor();
        ExprVisitor<StringBuilder> sv = new StringExprVisitor();
        Iterator<String> it = Pattern.compile(" ").splitAsStream("+ * 4 + 1 1 + 2 3").iterator();
        Expr expr = Expr.parseExpr(it);
        System.out.println(expr);
        System.out.println(expr.visit(sv).toString());
        System.out.println(expr.visit(v));
    }

}
