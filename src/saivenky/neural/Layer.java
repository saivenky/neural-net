package saivenky.neural;

import saivenky.neural.activation.ActivationFunction;
import saivenky.neural.neuron.NeuronInitializer;

/**
 * Created by saivenky on 1/26/17.
 */
class Layer {
    private double[] weightedInput;
    Neuron[] neurons;

    double[] activation;
    double[] activation1;
    private ActivationFunction activationFunction;

    double[] error;
    double[] previousLayerError;

    private double[] input;
    private double[] input1;

    Layer(
            int neuronCount, int previousLayerNeuronCount, ActivationFunction activationFunction, NeuronInitializer neuronInitializer) {
        weightedInput = new double[previousLayerNeuronCount];
        neurons = new Neuron[neuronCount];
        initializeNeurons(neuronInitializer, previousLayerNeuronCount);
        activation = new double[neuronCount];
        activation1 = new double[neuronCount];
        this.activationFunction = activationFunction;
        previousLayerError = new double[previousLayerNeuronCount];
    }

    private void initializeNeurons(
            NeuronInitializer neuronInitializer, int previousLayerNeuronCount) {
        for(int i = 0; i < neurons.length; i++) {
            neurons[i] = new Neuron(neuronInitializer, previousLayerNeuronCount);
        }
    }

    void run(double[] input, double[] input1) {
        this.input = input;
        this.input1 = input1;

        for(int i = 0; i < activation.length; i++) {
            double signal = neurons[i].signal(input, weightedInput);
            activation[i] = activationFunction.f(signal);
            activation1[i] = activationFunction.f1(signal);
        }
    }

    void backpropagate() {
        if(input1 != null) {
            for (int i = 0; i < neurons.length; i++) {
                Vector.multiplyAndAdd(neurons[i].weights, error[i], previousLayerError);
            }

            Vector.multiply(previousLayerError, input1, previousLayerError);
        }

        for(int i = 0; i < neurons.length; i++) {
            neurons[i].biasError += error[i];
            Vector.multiplyAndAdd(input, error[i], neurons[i].weightError);
        }
    }

    void update(double rate) {
        for (Neuron neuron : neurons) {
            neuron.update(rate);
        }
        Vector.zero(previousLayerError);
    }
}
