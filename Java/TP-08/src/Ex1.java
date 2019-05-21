class A {
    int x = 1;

    public int getX() {
        return x;
    }

    static void printX(A a) {
        System.out.println("in A.printX");
        System.out.println("x " + a.x);
        System.out.println("getX() " + a.getX());
    }

    int m(A a) { return 1; }
}

class B extends A {
    int x = 2;

    public int getX() {
        return x;
    }

    static void printX(B b) {
        System.out.println("in B.printX");
        System.out.println("x " + b.x);
        System.out.println("getX() " + b.getX());
    }

    int m(B b) { return 2; }
}

class Overridings {
    public static void main(String[] args) {
        // A.printX(new A());     /* Prévision : renvoie 1 => ok */
        //B.printX(new B());     /* Prévision : renvoie 2 => ok */
        //A.printX(new B());     /* Prévision : renvoie 2 => ok */
        A a = new B();
        System.out.println(a.m(a));   /* Prévision : renvoie 2 => non-ok, renvoie 1 car c'est la méthode qui prend un A en argument qui est appelée */
    }
}