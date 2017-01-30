package saivenky.neural.neuron;

import java.util.Random;

/**
 * Created by saivenky on 1/29/17.
 */
public class GaussianInitializer extends NeuronInitializer {
    private static Random random = new Random();
    public static Function gaussian = new Function() {
        @Override
        public double f() {
            return random.nextGaussian();
        }
    };

    public GaussianInitializer() {
        super(gaussian, gaussian);
    }
}
