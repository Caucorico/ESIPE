package fr.umlv.geom.main;

public class Main
{
  public static void main(String[] args)
  {
    Point p=new Point(1,2);
    Circle c=new Circle(p,2);
    System.out.println(c);
    Ring r = new Ring(p, 2, 1);
    System.out.println(r);
  }
}