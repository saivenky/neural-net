package saivenky.neural;

import saivenky.neural.activation.ActivationFunction;
import saivenky.neural.neuron.NeuronInitializer;

/**
 * Created by saivenky on 1/26/17.
 */
public class Neuron {
    final NeuronProperties properties;
    final NeuronSet inputNeurons;
    ActivationFunction activationFunction;
    public double activation;
    public double activation1;
    public double signalCostGradient;

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

    void signalForSelected() {
        double signal = properties.affineForSelected(inputNeurons);
        activation = activationFunction.f(signal);
        activation1 = activationFunction.f1(signal);
    }

    void signalScaled(double scale) {
        double signal = properties.affineScaled(inputNeurons, scale);
        activation = activationFunction.f(signal);
        activation1 = activationFunction.f1(signal);
    }

    void update(double rate) {
        properties.update(rate);
    }

    void addToSignalCostGradient(double weight, double cost) {
        signalCostGradient += weight * cost * activation1;
    }

    void propagateToInputNeurons() {
        inputNeurons.forSelected(new NeuronSet.NeuronAction() {
            @Override
            void f(Neuron neuron, int i) {
                neuron.addToSignalCostGradient(properties.weights[i], signalCostGradient);
            }
        });
    }

    void propagateToProperties() {
        properties.biasCostGradient += signalCostGradient;
        inputNeurons.forSelected(new NeuronSet.NeuronAction() {
            @Override
            void f(Neuron neuron, int i) {
                properties.weightCostGradient[i] += neuron.activation * signalCostGradient;
            }
        });
        signalCostGradient = 0;
    }
}
