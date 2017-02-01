package saivenky.neural;

import saivenky.neural.activation.ActivationFunction;

/**
 * Created by saivenky on 1/31/17.
 */
public class InputNeuronSet extends NeuronSet {
    public InputNeuronSet() {
        super();
    }

    void setInput(double[] input) {
        activation = input;
        selected = new int[input.length];
        Vector.select(selected, selected.length);
        neurons = new Neuron[0];
    }

    @Override
    public void activate(ActivationFunction activationFunction) {}

    @Override
    public void activateScaled(ActivationFunction activationFunction, double scale) {}

    @Override
    public void addSignalCostGradient(double[] cost, double scalar) {}

    @Override
    public void completeSignalCostGradient() {}

    @Override
    void clearSignalCostGradient() {}
}
