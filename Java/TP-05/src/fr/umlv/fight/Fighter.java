package fr.umlv.fight;

import java.util.Random;

/* Le sous-typage permet d'eviter a devoir recreer le code de Robot dans Fighter. On evite ainsi de dupliquer le code,
de devoir reecrir 2 fois le code si il est modifié, de recopier les erreurs et de devoir les corriger 2 fois.s
*/
public class Fighter extends Robot
{
    /* Une méthode pseudo aléatoire est une méthode qui renvoie un nombre obtenu par un algorithme en essayant de simuler
     * un comprtement aléatoire.
     * La graine est la valeur d'initialisation de cet algorithme. Par exemple, notre algo renvoie la n eme décimale de pi,
     * la graine permetterai de dire qu'on commence a compter a partir de la valeur de cette graine. L'algo renvoie dooc
     * la n+graine ieme decimale de pi.
     */

    /*
    * Il ne faut pas mettre les champs autre que protected ou de package pour éviter que notre objet soit modifiable
    * depuis l'exterieur et que son comportement diffère de celui prévu.
    */

    private Random r;

    public Fighter(String name, int seed)
    {
        super(name);
        this.r =  new Random(seed);
    }

    @Override
    public String toString()
    {
        return "Fighter "+this.getName();
    }

    @Override
    protected boolean rollDice()
    {
        return r.nextBoolean();
    }
}
