package saivenky.neural.activation;

/**
 * Created by saivenky on 1/27/17.
 */
public class Tanh implements ActivationFunction {
    private static ActivationFunction SINGLETON;

    public static ActivationFunction getInstance() {
        if (SINGLETON == null) {
            SINGLETON = new Tanh();
        }

        return SINGLETON;
    }

    private Tanh() {}

    @Override
    public double f(double z) {
        return Math.tanh(z);
    }

    @Override
    public double f1(double z) {
        double f = f(z);
        return 1 - f * f;
    }
}
