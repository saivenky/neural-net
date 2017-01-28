package saivenky.neural.cost;

import saivenky.neural.Vector;

/**
 * Created by saivenky on 1/28/17.
 */
public class CrossEntropy implements CostFunction {
    private static CostFunction SINGLETON;

    public static CostFunction getInstance() {
        if (SINGLETON == null) {
            SINGLETON = new CrossEntropy();
        }

        return SINGLETON;
    }

    private CrossEntropy() {}

    @Override
    public double f(double[] actual, double[] expected) {
        //y ln a + (1 - y) ln (1 - a)
        double[] result = new double[actual.length];
        for(int i = 0; i < result.length; i++) {
            double term1 = expected[i] * Math.log(actual[i]);
            double term2 = (1 - expected[i]) * Math.log(1 - actual[i]);
            result[i] = term1 + term2;
        }

        return -Vector.sum(result);
    }

    @Override
    public double[] f1(double[] actual, double[] expected) {
        double[] result = new double[actual.length];
        for(int i = 0; i < result.length; i++) {
            double term1 = expected[i] / actual[i];
            double term2 = (1 - expected[i]) / (1 - actual[i]);
            result[i] = term2 - term1;
        }
        return result;
    }
}
