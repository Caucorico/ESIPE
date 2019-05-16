package fr.umlv.geom;

public class Ring extends Circle
{
  private Circle voidArea;

  public Ring(Point center, int radius, int internRadius)
  {
    super(center, radius);
    if ( radius < internRadius )
    {
      throw new IllegalArgumentException("The intern radius is bigger than the extern");
    }
    this.voidArea = new Circle(center, internRadius);
  }

  @Override
  public String toString()
  {
    return super.toString() + " InternalRadius : " + this.voidArea.getRadius();
  }

  @Override
  public boolean contains(Point p)
  {
    return !this.voidArea.contains(p) && super.contains(p);
  }

  public static boolean contains(Point p, Ring... rings)
  {
    for( Ring r : rings )
    {
      if ( r.contains(p) ) return true;
    }

    return false;
  }
}