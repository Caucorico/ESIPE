package fr.umlv.fight;

public class Robot
{
    /**
     * The name of the robot
     */
    private String name;

    /**
     * The pv of the robot
     */
    private int life;

    /**
     * Robot Constructor.
     *
     * @param name The name of the robot.
     */
    public Robot(String name)
    {
        this.name = name;
        this.life = 10;
    }

    @Override
    public String toString()
    {
        return "Robot "+this.name;
    }

    /**
     * Return if the robot is dead or not.
     *
     * @return boolean
     */
    public boolean isDead()
    {
        if ( this.life <= 0 ) return true;
        return false;
    }


    /**
     * Shoot the robot argument with robot this.
     * This method can be static but it is for anticipate if the shoot depend of this.
     *
     * @param robot The robot target
     * @throws IllegalStateException If the robot is already dead, throw this exception
     */
    public void fire( Robot robot )
    {
        if ( robot.isDead() ) throw new IllegalStateException("The robot is dead ! ");

        if ( this.rollDice() )
        {
            robot.setLife( robot.getLife()-2 );
            System.out.println( robot.toString() +" a été touché par le "+ this.toString() +" !");
        }
        else
        {
            System.out.println( this.toString() + " a manqué le " + robot.toString() + " !");
        }
    }

    /* Cette methode doit etre protected pour que les sous-types puissent y acceder mais pas le reste */
    protected boolean rollDice()
    {
        return true;
    }

    public int getLife()
    {
        return life;
    }

    public void setLife(int life)
    {
        this.life = life;
    }

    public String getName()
    {
        return name;
    }
}
