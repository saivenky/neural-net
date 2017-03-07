package saivenky.neural;

import java.util.Random;

/**
 * Created by saivenky on 1/27/17.
 */
public class Data {
    public static class Example {
        public double[] input;
        public double[] output;


        public Example(double input[], double[] output) {
            this.input = input;
            this.output = output;
        }
    }

    private static final Random RANDOM = new Random();

    static <T> void shuffle(T[] array) {
        for(int i = 0; i < array.length; i++) {
            swap(array, i, RANDOM.nextInt(array.length - i) + i);
        }
    }

    private static <T> void swap(T[] array, int a, int b) {
        if(a == b) return;
        T temp = array[a];
        array[a] = array[b];
        array[b] = temp;
    }
}
