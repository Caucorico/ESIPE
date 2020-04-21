package fr.umlv.structconc;

import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class Vectorized {
    public static int sumLoop(int[] array) {
        var sum = 0;
        for(var value: array) {
            sum += value;
        }
        return sum;
    }

    private static final VectorSpecies<Integer> SPECIES = IntVector.SPECIES_PREFERRED;

    public static int sumReduceLane(int[] array) {
        var sum = 0;

        var i = 0;
        var limit = array.length - (array.length % SPECIES.length());  // main loop
        for (; i < limit; i += SPECIES.length()) {
            var vector = IntVector.fromArray(SPECIES, array, i);
            int result = vector.reduceLanes(VectorOperators.ADD);
            sum += result;
        }
        for (; i < array.length; i++) {                             // post loop
            // ne pas utiliser les vecteurs !
            sum += array[i];
        }

        return sum;
    }

    public static int sumLanewise(int[] array) {
        var vector = IntVector.zero(SPECIES);

        var i = 0;
        var limit = array.length - (array.length % SPECIES.length());  // main loop
        for (; i < limit; i += SPECIES.length()) {
            var subVector = IntVector.fromArray(SPECIES, array, i);
            vector = vector.lanewise(VectorOperators.ADD, subVector);
        }

        var sum = vector.reduceLanes(VectorOperators.ADD);

        for (; i < array.length; i++) {                             // post loop
            // ne pas utiliser les vecteurs !
            sum += array[i];
        }

        return sum;
    }
}
