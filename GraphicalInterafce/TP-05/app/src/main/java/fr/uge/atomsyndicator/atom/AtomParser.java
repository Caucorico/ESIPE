package fr.uge.atomsyndicator.atom;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.Serializable;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Stack;

import org.xmlpull.v1.XmlPullParser;
import org.xmlpull.v1.XmlPullParserException;
import org.xmlpull.v1.XmlPullParserFactory;

/**
 * A simple parser for Atom feeds.
 * 
 * @author chilowi at univ-mlv.fr
 *
 */
public class AtomParser 
{
	/** An atom entry */
	public static class Entry implements Serializable
	{
		public final String id;
		public final String title;
		public final Date date;
		public final String summary;
		public final String url;
		
		public Entry(String id, String title, Date date, String summary, String url)
		{
			this.id = id;
			this.title = title;
			this.date = date;
			this.summary = summary;
			this.url = url;
		}
		
		public static final int DIGESTED_SUMMARY_LENGTH = 16;
		@Override
		public String toString()
		{
			return String.format("[%s:%s@%s, %s, %s...]", id, title, date, url, (summary==null)?"":summary.substring(0, DIGESTED_SUMMARY_LENGTH));
		}

		@Override
		public boolean equals(Object obj)
		{
			if (! (obj instanceof Entry)) return false;
			return Objects.equals(((Entry)obj).id, this.id);
		}

		@Override
		public int hashCode()
		{
			return id.hashCode();
		}
	}
	
	XmlPullParser parser;
	
	public AtomParser(Reader r) throws XmlPullParserException
	{
		parser = XmlPullParserFactory.newInstance().newPullParser();
		parser.setInput(r);
	}
	
	// Some feeds specify milliseconds, other not
	public static final DateFormat[] DATE_FORMATS = new DateFormat[] {
		new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssX", Locale.US),
		new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSX", Locale.US)
	};
		
	// Try all the date formats to parse the date
	public static final Date parseDate(String s) throws ParseException
	{
		ParseException e = null;
		for (DateFormat format: DATE_FORMATS)
			try {
				return format.parse(s);
			} catch (ParseException e0)
			{
				e = e0;
			}
		throw e;
	}
	
	public boolean parse(List<Entry> entries) throws IOException
	{
		Stack<String> tagStack = new Stack<String>();
		String id = null, title = null, summary = null, link = null;
		Date date = null;
		for (boolean go = true; go;)
		{
			int tag = -1;
			try {
				tag = parser.next();
			} catch (XmlPullParserException e) { e.printStackTrace(); }
			switch (tag)
			{
			case XmlPullParser.START_TAG:
				String tagName = parser.getName();
				try {
					boolean push = false;
					if (tagName.equals("id"))
						id = parser.nextText();
					else if (tagName.equals("title"))
						title = parser.nextText();
					else if (tagName.equals("updated"))
						date = parseDate(parser.nextText());
					else if (tagName.equals("summary"))
						summary = parser.nextText();
					else if (tagName.equals("link"))
					{
						link = parser.getAttributeValue(null, "href");
						push = true;
					} else
						push = true;
					if (push)
					{
						tagStack.push(tagName);
					}
				} catch (XmlPullParserException e)
				{
					System.err.println("Parsing exception encountered: " + e);
					return false;
				} catch (ParseException e)
				{
					System.err.println(e);
					return false;
				}
				break;
			case XmlPullParser.END_TAG:
				String removed = tagStack.pop();
				if (! removed.equals(parser.getName()))
				{
					System.err.println(String.format("Encountered closing tag %s does not match the previously stacked tag %s", parser.getName(), removed));
					return false;
				}
				if (removed.equals("entry"))
				{
					entries.add(new Entry(id, title, date, summary, link));
					id = null; title = null; date = null; summary = null; link = null;
				}
				if (tagStack.isEmpty())
					go = false; // End of parsing
				break;
			case XmlPullParser.END_DOCUMENT:
				return false;
			default:
				break;
			}
		}
		return true;
	}
	
	public static void main(String[] args) throws Exception
	{
		if (args.length < 1) throw new IndexOutOfBoundsException("A file must be specified as the single argument");
		Reader r = new InputStreamReader(new FileInputStream(args[0]), "UTF-8");
		try {
			List<Entry> entries = new ArrayList<Entry>();
			new AtomParser(r).parse(entries);
			for (Entry entry: entries)
				System.out.println(entry);
		} finally
		{
			r.close();
		}
	}
}
