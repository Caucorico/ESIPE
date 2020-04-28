import java.util.ArrayList;
import java.util.Arrays;

/**
 *
 * @param <E>
 *
 *
 * Le code de notre premier CowList n'est pas thread-safe.
 * En effet, les méthodes add et size ne sont pas atomiques, elle peuvent être interrompues.
 * Un thread peut créer un nouveau tableau de taille 2 puis être interrompue
 * Un deuxième thread créé un tableau de taille 5, l'affecte à "tab".
 * Puis le premier thread reprend la main, et remet le tableau de taille 2.
 * Si le deuxième thread essaye d'ajouter un élément après ça, => ArrayOutOfBoundException.
 */
public class CowList<E> {

    private E[] tab;

    private static final Object[] EMPTY = new Object[0];

    @SuppressWarnings("unchecked")
    public CowList() {
        tab = (E[])EMPTY;
    }

    public void add(E e) {
        tab = Arrays.copyOf(tab, tab.length + 1);
        tab[tab.length - 1] = e;
    }

    public int size() {
        return tab.length;
    }

    public static void main(String[] args) throws InterruptedException {
        var threads = new ArrayList<Thread>();
        var cowList = new CowList<Integer>();

        for ( int i = 0 ; i < 4 ; i++ ) {
            var thread = new Thread(() -> {
                for ( int j = 0 ; j < 2_500 ; j++ ) {
                    cowList.add(j);
                }
            });
            thread.start();
            threads.add(thread);
        }

        for ( var thread : threads ) {
            thread.join();
        }

        System.out.println("Size = " + cowList.size());
    }
}
