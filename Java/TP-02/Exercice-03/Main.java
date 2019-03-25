import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class Main
{

	private int x;

	private int y;

	/*
		Les objets Patter et Matcher servent a evaluer des expression regulieres
	*/

	public static void displayIfNumber(String[] args)
	{
		Pattern p = Pattern.compile("^[^0-9]*([0-9]+)$");
		for ( String elem : args )
		{
			Matcher m = p.matcher(elem);
			if ( m.matches() )
			{	
				System.out.println( m.group(1) ); // 1 because it is the second group.
			}
		}
		
	}

	public static byte[] returnByteArrayOfIpv4(String ipv4)
	{
		Pattern p = Pattern.compile("^([0-9]+).([0-9]+).([0-9]+).([0-9]+)$");
		Matcher m = p.matcher(ipv4);
		byte[] bt = new byte[4];

		if ( !m.matches() ) return null;
		for ( int i = 1 ; i <= 4 ; i++ )
		{
			if( Integer.parseInt(m.group(i)) < 0 ||  Integer.parseInt(m.group(i)) > 255  )
			{
				return null;
			}
			else
			{
				int test = Integer.parseInt(m.group(i))&255;
				bt[i-1] = (byte)test;
			}
		}
		return bt;
	}

	public static void main(String[] args)
	{
		byte[] res = Main.returnByteArrayOfIpv4(args[0]);
		if ( res == null ) System.out.println("bad format");
		else
		{
			for ( int i = 0 ; i < res.length ; i++ )
			{
				System.out.println(res[i]&0xff);
			}
		}
	}


}