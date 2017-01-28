package saivenky.neural;

import java.util.Random;

/**
 * Created by saivenky on 1/27/17.
 */
public class Data {
    public static class Example {
        double[] input;
        double[] output;


        public Example(double input[], double[] output) {
            this.input = input;
            this.output = output;
        }

        public static Example generateXor() {
            double a = uniform(-5, 5);
            double b = uniform(-5, 5);
            double product = a * b;
            product = (product > 0) ? 1 : -1;
            return new Example(Vector.ize(a, b), Vector.ize(product));
        }
    }

    private static final Random RANDOM = new Random();

    public static double uniform(double min, double max) {
        return (max - min) * RANDOM.nextDouble() + min;
    }

    public static Example[] generateXor(int sampleSize) {
        Example[] samples = new Example[sampleSize];
        for (int i = 0; i < sampleSize; i++) {
            samples[i] = Example.generateXor();
        }

        return samples;
    }
}
