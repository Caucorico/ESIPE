package fr.umlv.geom;

public class Circle
{
  /* Ce champ doit être private pour n'être modifiable que depuis un Cercle*/
  private Point center;

  /* Ce champ doit être private pour n'être modifiable que depuis un Cercle*/
  private int radius;

  public Circle(Point center, int radius)
  {
    this.center = center;
    this.radius = radius;
  }

  @Override
  public String toString()
  {
    return "Cercle de rayon " + this.radius + ", de centre " + this.center.toString()
      + " et de surface " + this.surface() + "\n";
  }

  /* Avec cette méthode, si l'on modifie les coordonnées du centre d'un cercle,
     si le point était déja utilisé, le deuxième cercle à lui aussi changé de coordonnées.


  public void translate(int dx, int dy)
  {
    this.center.translate(dx, dy);
  } */

  public void translate(int dx, int dy)
  {
    Point new_center = new Point(center.getX(), center.getY());
    new_center.translate(dx, dy);
    this.center = new_center;
  }

  /* Avec cette méthode, on peux obtenir le point via l'accesseur et le modifier et ainsi créer des effets de bords
  public Point getCenter()
  {
    return this.center;
  }*/

  public Point getCenter()
  {
    return new Point(this.center.getX(), this.center.getY());
  }

  public int getRadius()
  {
    return this.radius;
  }

  public double surface()
  {
    return Math.PI * Math.pow( this.radius, 2 );
  }

  public boolean contains(Point p)
  {
    return Math.sqrt(
      Math.pow(p.getX() - this.center.getX(), 2 )
      + Math.pow(p.getY() - this.center.getY(), 2 )
      ) <= this.radius;
  }
}