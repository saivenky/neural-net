package saivenky.neural.activation;

/**
 * Created by saivenky on 1/25/17.
 */
public class Sigmoid implements ActivationFunction {
    private static ActivationFunction SINGLETON;

    public static ActivationFunction getInstance() {
        if (SINGLETON == null) {
            SINGLETON = new Sigmoid();
        }

        return SINGLETON;
    }

    private Sigmoid() {}

    @Override
    public double f(double z) {
        return 1. / (1. + Math.exp(-z));
    }

    @Override
    public double f1(double z) {
        double f = f(z);
        return f * (1. - f);
    }
}
