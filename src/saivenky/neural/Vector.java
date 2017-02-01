package saivenky.neural;

import java.util.HashMap;
import java.util.Random;

/**
 * Created by saivenky on 1/26/17.
 */
public class Vector {
    private static Random random = new Random(0);

    public static void initialize(long seed) {
        random = new Random(seed);
    }

    private static void checkSizes(double[] ... arrays) {
        int length = arrays[0].length;
        for(double[] a : arrays) {
            if (a == null || a.length != length) {
                throw new RuntimeException("Vector length mismatch");
            }
        }
    }

    public static void multiply(double[] a, double[] b, double[] product) {
        checkSizes(a, b, product);
        for(int i = 0; i < product.length; i++) {
            product[i] = a[i] * b[i];
        }
    }

    public static void multiplySelected(double[] a, double[] b, double[] product, int[] selected) {
        if (selected == null) {
            multiply(a, b, product);
            return;
        }

        checkSizes(a, b, product);
        for (int i : selected) {
            product[i] = a[i] * b[i];
        }
    }

    public static void multiplyAndAdd(double[] a, double s, double[] result) {
        checkSizes(a, result);
        for(int i = 0; i < result.length; i++) {
            result[i] += a[i] * s;
        }
    }

    public static double sum(double[] vector) {
        double sum = 0;
        for(double a : vector) {
            sum += a;
        }

        return sum;
    }

    public static double sumSelected(double[] vector, int[] selected) {
        double sum = 0;
        for(int i : selected) {
            sum += vector[i];
        }

        return sum;
    }

    public static void zero(double[] vector) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] = 0;
        }
    }

    public static void subtract(double[] a, double[] b, double[] result) {
        checkSizes(a, b, result);
        for(int i = 0; i < result.length; i++) {
            result[i] = a[i] - b[i];
        }
    }

    static String str(double[] a) {
        StringBuilder builder = new StringBuilder();
        builder.append('[');
        for(int i = 0; i < a.length; i++) {
            builder.append(a[i]);
            if ((i + 1) < a.length) builder.append(", ");
        }
        builder.append(']');
        return builder.toString();
    }

    static String str(int[] a) {
        StringBuilder builder = new StringBuilder();
        builder.append('[');
        for(int i = 0; i < a.length; i++) {
            builder.append(a[i]);
            if ((i + 1) < a.length) builder.append(", ");
        }
        builder.append(']');
        return builder.toString();
    }

    public static void print(String label, double[] a) {
        System.out.printf("%s: %s\n", label, str(a));
    }
    public static void print(String label, int[] a) {
        System.out.printf("%s: %s\n", label, str(a));
    }

    public static double sumSquared(double[] a) {
        double sumSquared = 0;
        for(int i = 0; i < a.length; i++) {
            sumSquared += a[i] * a[i];
        }

        return sumSquared;
    }

    public static int[] select(int[] selected, int length) {
        int needed = selected.length;
        int remaining = length;
        for(int i = 0; i < length; i++) {
            if(isSelected((double)needed/remaining)) {
                selected[selected.length - needed--] = i;
            }

            remaining--;
        }

        return selected;
    }

    private static boolean isSelected(double selectionProbability) {
        return random.nextDouble() < selectionProbability;
    }

    public static double[] ize(double ... nums) {
        return nums;
    }

    public static void main(String[] args) {
        Vector.initialize(System.currentTimeMillis());
        int[] selected = new int[3];
        for(int i = 0; i < 10; i++) {
            select(selected, 10);
            print("selected", selected);
        }
    }

    public static void multiplyAndAddSelected(double[] a, double s, double[] result, int[] selected) {
        if (selected == null) {
            multiplyAndAdd(a, s, result);
            return;
        }

        checkSizes(a, result);
        for(int i : selected) {
            result[i] += a[i] * s;
        }
    }

    private static HashMap<Integer, double[]> temporaryVectors = new HashMap<>();
    public static double[] getTemporaryVector(int length) {
        if(!temporaryVectors.containsKey(length)) {
            temporaryVectors.put(length, new double[length]);
        }

        return temporaryVectors.get(length);
    }
}
