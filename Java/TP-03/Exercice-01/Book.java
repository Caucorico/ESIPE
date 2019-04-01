class Book
{
	private String title;
	private String author;

	/* Aucun probleme lors du changement des arguments. Cependant, si j'avais omis de mettre this, il y aurait eu conflit .*/
	public Book(String title, String author)
	{
		this.title = title;
		this.author = author;
	}

	/* Le compilateur regarde avec quels arguments on utilise le constructeur
	 * et appelle celui qui correspond et refuse de compiler si il ne le trouve pas.
	 */
	/* Pour que ce constructeur utilise le premier, il faut l'appeler avec les bon arguments et this... */
	public Book(String title)
	{
		this(title, "<no author>");
	}

	/*
	 * Le code du main ne fonctionne plus car le constructeur qui ne prends aucun argument n'existe plus.
	 * Il faut donc ajouter un titre et un atheur a chaque nouveau livre cree
	 */
	public static void main(String[] args)
	{
		Book book = new Book("toto", "tata");

		/* Les attributs title et author etant private, on peut s'attendre a voir une erreur,
		 * nous n'avons pas le droit d'acceder a ces attributs en dehors de l'objet. 
		 */
	    System.out.println(book.title + ' ' + book.author);
	    /* Aucune erreur de compilation, on pouvait s'attendre a voir une erreur de compilation. */
	    /* Apres execution, les champs suivant n'ont pas leve d'exception et renvoie null,
	     * conclusion : faire attention
	     */
	}
}