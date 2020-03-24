package fr.umlv.info2.graphs;

public class Edge {
    private final int start;
    private final int end;
    private final int value;

    public Edge(int start, int end, int value) {
        super();
        this.start = start;
        this.end = end;
        this.value = value;
    }

    public Edge(int start, int end) {
        this(start, end, 1);
    }

    public int getValue() {
        return value;
    }

    public int getStart() {
        return start;
    }

    public int getEnd() {
        return end;
    }

    @Override
    public String toString() {
        return start + " -- " + end + " ( " + value + " ) ";
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + end;
        result = prime * result + start;
        result = prime * result + value;
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        Edge other = (Edge) obj;
        if (end != other.end)
            return false;
        if (start != other.start)
            return false;
        if (value != other.value)
            return false;
        return true;
    }

}