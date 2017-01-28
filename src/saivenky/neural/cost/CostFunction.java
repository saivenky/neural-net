package saivenky.neural.cost;

/**
 * Created by saivenky on 1/27/17.
 */
public interface CostFunction {
    double f(double[] actual, double[] expected);
    double[] f1(double[] actual, double[] expected);
}
