package fr.umlv.conc;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.file.Path;
import java.util.ArrayList;

public class Utils {
    private static Path HOME;

    private static Path HOME1;
    private static final Object lock = new Object();

    private static volatile Path HOME2;

    private static Path HOME3;
    private static final VarHandle HANDLE_HOME3;

    static {
        try {
            HANDLE_HOME3 = MethodHandles.lookup().findStaticVarHandle(Utils.class, "HOME3", Path.class);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new AssertionError();
        }
    }

    public static Path getHome1() {
        synchronized (lock) {
            if (HOME1 == null) {
                return HOME1 = Path.of(System.getenv("HOME"));
            }
            return HOME1;
        }
    }

    public static Path getHome2() {
        var home = HOME2;
        if (home == null) {
            synchronized(Utils.class) {
                home = HOME2;
                if (home == null) {
                    return HOME2 = Path.of(System.getenv("HOME"));
                }
            }
        }
        return home;
    }

    private static int cpt3 = 0;
    public static Path getHome3() {
        Path home = (Path)HANDLE_HOME3.getAcquire();
        if (home == null) {
            synchronized(Utils.class) {
                home = (Path)HANDLE_HOME3.getAcquire();
                if (home == null) {
                    home = Path.of(System.getenv("HOME"));
                    HANDLE_HOME3.setRelease(home);
                    cpt3++;
                }
            }
        }

        return home;
    }

    private static class Holder {
        static final Path HOME = Path.of(System.getenv("HOME"));
    }
    public static Path getHome4() {
        return Holder.HOME;
    }

    public static void main(String[] args) throws InterruptedException {
        ArrayList<Thread> threads = new ArrayList<>(10);

        for ( var i = 0 ; i < 10 ; i++ ) {
            Thread t = new Thread(() -> {
                var home = getHome4();
                System.out.println(home);
            });
            threads.add(t);
            t.start();
        }

        for ( var i = 0 ; i < 10 ; i++ ) {
            threads.get(i).join();
        }
    }

}
