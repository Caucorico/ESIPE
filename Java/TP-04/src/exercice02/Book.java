package exercice02;

import java.util.Objects;

class Book
{
	private final String title;
	private final String author;
	private final boolean hasAuthor;

	public Book(String title, String author)
	{
		this.title = Objects.requireNonNull(title);
		this.author = Objects.requireNonNull(author);
		this.hasAuthor = true;
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

	/* Peut-on utiliser l'annotation @Override, ici ? 
	 * Oui, car toString est une methode de Object et tout objet herite de Object.
	 */
	@Override
	public String toString()
	{
		if ( this.hasAuthor )
		{
			return this.title+" by "+this.author;
		}
		else
		{
			return this.title;
		}
		
	}
}