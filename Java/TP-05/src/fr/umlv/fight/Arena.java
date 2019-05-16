package fr.umlv.fight;

public class Arena
{
    public static Robot fight(Robot r1, Robot r2)
    {
        int turn = 0;

        /* Le polymorphisme est utile ici car que ce soit un robot ou un fighter, les methodes a appeler sont les memes
         * Et le fait que le Robot soit un robot ou non n'est pas important ici.
         */
        while ( !r1.isDead() && !r2.isDead() )
        {
            if ( turn%2 == 0 )
            {
                r1.fire(r2);
            }
            else
            {
                r2.fire(r1);
            }
            turn++;
        }

        if ( r1.isDead() ) return r2;
        else return r1;
    }

    public static void main(String[] args)
    {
        Fighter john = new Fighter("John", 1);
        Fighter jane = new Fighter("Jane", 2);
        System.out.println(fight(john, jane) + " wins");
    }
}
