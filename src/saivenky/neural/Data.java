package saivenky.neural;

import java.util.Random;

/**
 * Created by saivenky on 1/27/17.
 */
public class Data {
    public static abstract class Function {
        abstract double f(double x);
    }
    public static class Example {
        public double[] input;
        public double[] output;


        public Example(double input[], double[] output) {
            this.input = input;
            this.output = output;
        }

        public static Example generateXor() {
            double a = uniform(-5, 5);
            double b = uniform(-5, 5);
            double product = a * b;

            product = binary(product);
            return new Example(Vector.ize(a, b), Vector.ize(product));
        }

        private static double binary(double num) {
            return (num < 0) ? 0 : 1;
        }

        private static double greaterThanFunction(Function function, double x, double y) {
            return binary(y - function.f(x));
        }

        public static Example generateFunction(Function function) {
            double a = uniform(-5, 5);
            double b = uniform(-5, 5);
            double label = greaterThanFunction(function, a, b);

            return new Example(Vector.ize(a, b), Vector.ize(label));
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

    public static Example[] generateFunction(Function function, int sampleSize) {
        Example[] samples = new Example[sampleSize];
        for (int i = 0; i < sampleSize; i++) {
            samples[i] = Example.generateFunction(function);
        }

        return samples;
    }

    public static <T> void shuffle(T[] array) {
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
