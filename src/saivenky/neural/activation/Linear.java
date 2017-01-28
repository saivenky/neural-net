package saivenky.neural.activation;

/**
 * Created by saivenky on 1/27/17.
 */
public class Linear implements ActivationFunction {
    private static ActivationFunction SINGLETON;

    public static ActivationFunction getInstance() {
        if (SINGLETON == null) {
            SINGLETON = new Linear();
        }

        return SINGLETON;
    }

    @Override
    public double f(double z) {
        return z;
    }

    @Override
    public double f1(double z) {
        return 1;
    }
}
