package saivenky.neural.neuron;

import java.util.Random;

/**
 * Created by saivenky on 1/29/17.
 */
public class GaussianInitializer extends NeuronInitializer {
    private static NeuronInitializer SINGLETON;
    private static Random random = new Random();
    private static Function gaussian = new Function() {
        @Override
        public double f() {
            return random.nextGaussian();
        }
    };

    public static NeuronInitializer getInstance() {
        if (SINGLETON == null) {
            SINGLETON = new GaussianInitializer();
        }

        return SINGLETON;
    }

    private GaussianInitializer() {
        super(gaussian, gaussian);
    }
}
