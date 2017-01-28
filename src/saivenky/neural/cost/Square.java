package saivenky.neural.cost;

import saivenky.neural.Vector;

/**
 * Created by saivenky on 1/27/17.
 */
public class Square implements CostFunction {
    private static CostFunction SINGLETON;

    public static CostFunction getInstance() {
        if (SINGLETON == null) {
            SINGLETON = new Square();
        }

        return SINGLETON;
    }

    private Square() {}

    @Override
    public double f(double[] actual, double[] expected) {
        double[] result = new double[actual.length];
        Vector.subtract(actual, expected, result);
        return 0.5 * Vector.sumSquared(result);
    }

    @Override
    public double[] f1(double[] actual, double[] expected) {
        double[] result = new double[actual.length];
        Vector.subtract(actual, expected, result);
        return result;

    }
}
