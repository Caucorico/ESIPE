import java.util.Objects;

class Book
{
	private String title;
	private String author;

	public Book(String title, String author)
	{
		this.title = Objects.requireNonNull(title);
		this.author = Objects.requireNonNull(author);
	}

	public Book(String title)
	{
		this(Objects.requireNonNull(title), "<no author>");
	}

	public String getTitle()
	{
		return this.title;
	}

	public String getAuthor()
	{
		return this.author;
	}

	/* L'annotation @Override signifie que l'on souhaite redefinir une methode existante et on demande donc au compilateur de verifier. */
	@Override
	public boolean equals(Object book)
	{
		if ( book instanceof Book )
		{
			if ( this.title.equals( ((Book)book).getTitle() )
				&& this.author.equals( ((Book)book).getAuthor() ) )
			{
				return true;
			}
			else
			{
				return false;
			}
		}
		else
		{
			return false;
		}
	}

	/* A quoi sert la méthode indexOf de ArrayList (RTFM) ?
	 * Cette methode sert a retourner l'index (la position) de l'objet passe en argument dans l'ArrayList ou -1 si il ne le trouve pas.
	 */

}