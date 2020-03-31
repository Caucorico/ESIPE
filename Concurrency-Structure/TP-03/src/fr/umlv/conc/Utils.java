package fr.umlv.conc;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.file.Path;

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

    public static Path getHome3() {
        var home = HANDLE_HOME3.getAcquire();
        if (home == null) {
            synchronized(Utils.class) {
                home = HOME; // TODO ??
                if (home == null) {
                    return HOME = Path.of(System.getenv("HOME"));  // TODO ??
                }
            }
        }
        return home;
    }

}
