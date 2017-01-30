package saivenky.neural.neuron;

import java.util.Random;

/**
 * Created by saivenky on 1/29/17.
 */
public class ZeroInitializer extends NeuronInitializer {
    public static Function zero = new Function() {
        @Override
        public double f() {
            return 0;
        }
    };

    public ZeroInitializer() {
        super(zero, zero);
    }
}
