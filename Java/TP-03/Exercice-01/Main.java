class Main
{
	public static void main(String[] args)
	{
		Book book = new Book();
		
		/* Cette ligne provoque une erreur de compilation.
		 * En effet, les attributs private de la classe sont inaccessible depuis l'exterieur.
		 */
	    System.out.println(book.title + ' ' + book.author);

	    /* Pour corriger ce probleme, nous avons besoin de creer des getteurs. Il ne faut cependant pas mettre les attributs en publics */

	    /* Les 4 visibilites possible en Java sont : public, package friendly(default), protected et private */
	    /* Les attributes doivent toujours etre declares en private pour eviter au maximum les effets de bords. */

	    /* Un accesseur est un getter, il renvoie l'attribut demande */
	    /* Un mutateur est un setteur, il modifie l'attribut demande et renvoie this si possible. */
	    /* Il faut ajouter les accesseur getTitle() et getAuthor() */

	    /* Pour indiquer a un futur developpeur que les champs title et author ne doivent pas etre modifie,
	     * il faut ajouter le mot clef final devant la variable
	     * Il serait aussi pertinant d'ajouter de la documentation pour prevenir les autres developpeur et de verifier que
	     * ces champs n'ont pas ete initialise a null.
		 *
         * Il est important de le faire pour qu'un autre programmeur ou nous meme voie les erreurs adequatent et puisse les corriger.
	     */

	}
}