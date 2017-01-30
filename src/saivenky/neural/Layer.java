package saivenky.neural;

import saivenky.neural.activation.ActivationFunction;
import saivenky.neural.neuron.NeuronInitializer;

/**
 * Created by saivenky on 1/26/17.
 */
public class Layer {
    double[] weightedInput;
    Neuron[] neurons;

    double[] activation;
    double[] activation1;
    ActivationFunction activationFunction;

    double[] error;
    double[] previousLayerError;

    double[] input;
    double[] input1;

    public Layer(
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

    public void run(double[] input, double[] input1) {
        this.input = input;
        this.input1 = input1;

        for(int i = 0; i < activation.length; i++) {
            double signal = neurons[i].signal(input, weightedInput);
            activation[i] = activationFunction.f(signal);
            activation1[i] = activationFunction.f1(signal);
        }
    }

    public void backpropagate() {
        if(input1 != null) {
            for (int i = 0; i < neurons.length; i++) {
                Vector.multiplyAndAdd(neurons[i].weights, error[i], previousLayerError);
            }

            Vector.multiply(previousLayerError, input1, previousLayerError);
        }

        for(int i = 0; i < neurons.length; i++) {
            Vector.multiplyAndAdd(input, error[i], neurons[i].weightError);
            neurons[i].biasError += error[i];
        }
    }

    public void update(double rate) {
        for(int i = 0; i < neurons.length; i++) {
            neurons[i].update(rate);
        }
        Vector.zero(previousLayerError);
    }
}
