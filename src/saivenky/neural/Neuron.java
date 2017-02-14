package saivenky.neural;

import saivenky.neural.activation.ActivationFunction;
import saivenky.neural.neuron.NeuronInitializer;

/**
 * Created by saivenky on 1/26/17.
 */
public class Neuron implements INeuron {
    final NeuronProperties properties;
    private final NeuronSet inputNeurons;
    private final ActivationFunction activationFunction;
    private double activation;
    private double activation1;
    private double signalCostGradient;

    Neuron(NeuronProperties properties, NeuronSet inputNeurons, ActivationFunction activationFunction) {
        this.properties = properties;
        this.inputNeurons = inputNeurons;
        this.activationFunction = activationFunction;
    }

    Neuron(NeuronInitializer neuronInitializer, NeuronSet inputNeurons, ActivationFunction activationFunction) {
        this.inputNeurons = inputNeurons;
        this.activationFunction = activationFunction;
        properties = new NeuronProperties(neuronInitializer, inputNeurons.size());
    }

    @Override
    public double getActivation() {
        return activation;
    }

    public void activate() {
        double signal = properties.affineForSelected(inputNeurons);
        activation = activationFunction.f(signal);
        activation1 = activationFunction.f1(signal);
    }

    public void activateScaled(double scale) {
        double signal = properties.affine(inputNeurons);
        activation = activationFunction.f(signal) * scale;
        activation1 = 0;
    }

    public void gradientDescent(double rate) {
        properties.update(rate);
    }

    @Override
    public void setSignalCostGradient(double cost) {
        signalCostGradient = cost;
    }

    @Override
    public void addToSignalCostGradient(double cost) {
        signalCostGradient += cost;
    }

    public synchronized void addToSignalCostGradient(double weight, double cost) {
        signalCostGradient += weight * cost;
    }

    private void backpropagateToInputNeurons() {
        inputNeurons.addSignalCostGradient(properties.getWeights(), signalCostGradient);
    }

    private void updateGradient() {
        properties.addError(inputNeurons, signalCostGradient);
    }

    public void backpropagate(boolean backpropagateToInputNeurons) {
        signalCostGradient *= activation1;
        updateGradient();
        if(backpropagateToInputNeurons) backpropagateToInputNeurons();
        signalCostGradient = 0;
    }
}
