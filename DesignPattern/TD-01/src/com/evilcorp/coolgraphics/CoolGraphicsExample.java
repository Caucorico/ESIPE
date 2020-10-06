package com.evilcorp.coolgraphics;

import com.evilcorp.coolgraphics.CoolGraphics.ColorPlus;


public class CoolGraphicsExample {

    public static void main(String[] args) {
            CoolGraphics area=new CoolGraphics("Example",800,600);
            area.repaint(ColorPlus.WHITE);
            // il n'y a pas de méthode pour tracer un rectangle, il faut le faire à la main
            area.drawLine(100,20,140, 20, ColorPlus.BLACK);
            area.drawLine(100,160,140, 160, ColorPlus.BLACK);
            area.drawLine(100,20,100, 160, ColorPlus.BLACK);
            area.drawLine(140,20,140, 160, ColorPlus.BLACK);

            area.drawLine(100, 20, 140, 160, ColorPlus.BLACK);
            area.drawLine(100, 160, 140, 20, ColorPlus.BLACK);

            area.drawEllipse(100, 20, 40, 140,ColorPlus.BLACK);

            area.waitForMouseEvents((x,y) -> System.out.println("Click on ("+x+","+y+")"));
    }

}
