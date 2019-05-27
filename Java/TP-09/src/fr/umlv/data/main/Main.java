package fr.umlv.data.main;

import fr.umlv.data.LinkedList;

public class Main
{
    public static void main(String[] args)
    {
        /*LinkedList lk = new LinkedList();

        lk.add(13);
        lk.add(144);

        System.out.println(lk.toString());*/

        LinkedList<String> lk = new LinkedList<>();

        lk.add("hello");
        lk.add("world");

        System.out.println(lk.toString());

        /* Nous sommes obligés d'utiliser un cast sur la ligne ci-dessous car on ne peut pas acceder a la fonction
         * length d'un objet String si celui-ci est vu en tant qu'object.
         * En java, nous n'aimons pas les cast car on peux ne pas detecter une erreur rapidement dans certains cas.
         * On peut se rendre compte trop tard que l'objet que l'on souhaite n'est pas du type voulu !
         */
        System.out.println("size = " + lk.get(1).length());

        System.out.println("contain toto ? : " + lk.contains("toto"));
        System.out.println("contain word ? : " + lk.contains("world"));
    }
}
