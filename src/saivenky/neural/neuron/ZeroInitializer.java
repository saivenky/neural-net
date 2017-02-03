package saivenky.neural.neuron;

/**
 * Created by saivenky on 1/29/17.
 */
public class ZeroInitializer extends NeuronInitializer {
    private static NeuronInitializer SINGLETON;

    public static Function zero = new Function() {
        @Override
        public double f() {
            return 0;
        }
    };

    public static NeuronInitializer getInstance() {
        if (SINGLETON == null) {
            SINGLETON = new ZeroInitializer();
        }

        return SINGLETON;
    }

    public ZeroInitializer() {
        super(zero, zero);
    }
}
