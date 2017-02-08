package saivenky.neural;

import java.util.Random;

/**
 * Created by saivenky on 1/26/17.
 */
public class Vector {
    private static Random random = new Random();

    private static void checkSizes(double[] ... arrays) {
        int length = arrays[0].length;
        for(double[] a : arrays) {
            if (a == null || a.length != length) {
                throw new RuntimeException("Vector length mismatch");
            }
        }
    }

    static void multiplyAndAdd(double[] a, double s, double[] result) {
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

    static void zero(double[] vector) {
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

    public static double sumSquared(double[] array) {
        double sumSquared = 0;
        for (double num : array) {
            sumSquared += num * num;
        }

        return sumSquared;
    }

    static int[] select(int[] selected, int length) {
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

    static double[] ize(double... nums) {
        return nums;
    }
}
