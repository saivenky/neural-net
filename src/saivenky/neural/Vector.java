package saivenky.neural;

import java.util.Random;

/**
 * Created by saivenky on 1/26/17.
 */
public class Vector {
    private static Random random = new Random(0);

    public static void initialize(long seed) {
        random = new Random(seed);
    }

    public static void multiply(double[] a, double[] b, double[] product) {
        for(int i = 0; i < product.length; i++) {
            product[i] = a[i] * b[i];
        }
    }

    public static void multiplyAndAdd(double[] a, double s, double[] result) {
        for(int i = 0; i < result.length; i++) {
            result[i] += a[i] * s;
        }
    }

    public static void add(double[] a, double[] b, double[] sum) {
        for(int i = 0; i < sum.length; i++) {
            sum[i] = a[i] + b[i];
        }
    }

    public static void random(double[] a) {
        for(int i = 0; i < a.length; i++) {
            a[i] = random.nextGaussian();
        }
    }

    public static double sum(double[] vector) {
        double sum = 0;
        for(double a : vector) {
            sum += a;
        }

        return sum;
    }

    public static void zero(double[] vector) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] = 0;
        }
    }

    public static void subtract(double[] a, double[] b, double[] result) {
        for(int i = 0; i < result.length; i++) {
            result[i] = a[i] - b[i];
        }
    }

    public static String str(double[] a) {
        StringBuilder builder = new StringBuilder();
        builder.append('[');
        for(int i = 0; i < a.length; i++) {
            builder.append(a[i]);
            if ((i + 1) < a.length) builder.append(", ");
        }
        builder.append(']');
        return builder.toString();
    }

    public static void print(double[] a) {
        System.out.println(str(a));
    }

    public static double sumSquared(double[] a) {
        double sumSquared = 0;
        for(int i = 0; i < a.length; i++) {
            sumSquared += a[i] * a[i];
        }

        return sumSquared;
    }

    public static double[] ize(double ... nums) {
        return nums;
    }
}
