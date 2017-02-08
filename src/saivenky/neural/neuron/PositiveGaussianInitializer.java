package saivenky.neural.neuron;

import java.util.Random;

/**
 * Created by saivenky on 2/8/17.
 */
public class PositiveGaussianInitializer extends NeuronInitializer {
    private static NeuronInitializer SINGLETON;
    private static Random random = new Random();
    private static Function gaussian = new Function() {
        @Override
        public double f() {
            return Math.abs(random.nextGaussian());
        }
    };

    public static NeuronInitializer getInstance() {
        if (SINGLETON == null) {
            SINGLETON = new PositiveGaussianInitializer();
        }

        return SINGLETON;
    }

    private PositiveGaussianInitializer() {
        super(gaussian, gaussian);
    }
}
