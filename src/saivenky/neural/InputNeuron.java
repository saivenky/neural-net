package saivenky.neural;

/**
 * Created by saivenky on 1/31/17.
 */
public class InputNeuron extends Neuron {

    public InputNeuron() {
        super((NeuronProperties)null, null, null);
    }

    void setActivation(double activation) {
        this.activation = activation;
    }

    @Override
    void signalForSelected() {}

    @Override
    void signalScaled(double scale) {}

    @Override
    void update(double rate) {}

    @Override
    void addToSignalCostGradient(double weight, double cost) {}

    @Override
    void propagateToInputNeurons() {}

    @Override
    void propagateToProperties() {}
}
