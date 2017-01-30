package saivenky.neural.neuron;

/**
 * Created by saivenky on 1/28/17.
 */
public class NeuronInitializer {
    private Function biasInitializer;
    private Function weightInitializer;

    public static abstract class Function {
        public abstract double f();
    }

    public NeuronInitializer(Function biasInitializer, Function weightInitializer) {
        this.biasInitializer = biasInitializer;
        this.weightInitializer = weightInitializer;
    }

    public double createBias() {
        return biasInitializer.f();
    }

    public double createWeight() {
        return weightInitializer.f();
    }
}
