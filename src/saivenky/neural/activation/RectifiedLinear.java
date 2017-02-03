package saivenky.neural.activation;

/**
 * Created by saivenky on 2/1/17.
 */
public class RectifiedLinear implements ActivationFunction{
    private static ActivationFunction SINGLETON;

    public static ActivationFunction getInstance() {
        if (SINGLETON == null) {
            SINGLETON = new RectifiedLinear();
        }

        return SINGLETON;
    }

    private RectifiedLinear() {}

    @Override
    public double f(double z) {
        return Math.max(0, z);
    }

    @Override
    public double f1(double z) {
        return (z < 0) ? 0 : 1;
    }
}
