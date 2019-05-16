package fr.umlv.geom;

public class Point {
	private int x;
	private int y;

	public Point(int x, int y) {
		this.x = x;
		this.y = y;
	}

	public int getX() {
		return x;
	}

	public int getY() {
		return y;
	}

	@Override
	public String toString() {
		return "(" + x + ',' + y + ')';
	}
	
	/* Cette méthode de compile pas car les variables x et y sont final, elle ne peuvent pas être modifié. 
	 * Si on laisse les parametres en final, le point ne pourra pas être dépacé par la méthode translate.
	 * Il faudrait créer un nouveau point avec les informations de l'ancien point, dx et dy.
	 */
	public void translate(int dx, int dy) {
	  x += dx;
	  y += dy;
	}
}
