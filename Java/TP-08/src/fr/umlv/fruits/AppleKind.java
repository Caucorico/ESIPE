package fr.umlv.fruits;

public enum AppleKind
{
    Golden, Pink_Lady, Granny_Smith;

    @Override
    public String toString()
    {
        return this.name().replace('_', ' ');
    }
}
