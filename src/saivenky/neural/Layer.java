package saivenky.neural;

import saivenky.neural.activation.ActivationFunction;
import saivenky.neural.activation.Sigmoid;
import saivenky.neural.activation.Tanh;

/**
 * Created by saivenky on 1/26/17.
 */
public class Layer {
    double[] bias;
    double[] weightedInput;
    Neuron[] neurons;

    double[] activation;
    double[] activation1;
    ActivationFunction activationFunction;

    double[] error;
    double[] previousLayerError;

    public Layer(int neuronCount, int previousLayerNeuronCount) {
        bias = new double[neuronCount];
        weightedInput = new double[previousLayerNeuronCount];
        neurons = new Neuron[neuronCount];
        initializeNeurons(previousLayerNeuronCount);
        activation = new double[neuronCount];
        activation1 = new double[neuronCount];
        activationFunction = Sigmoid.getInstance();
        previousLayerError = new double[previousLayerNeuronCount];
    }

    private void initializeNeurons(int previousLayerNeuronCount) {
        for(int i = 0; i < neurons.length; i++) {
            neurons[i] = new Neuron(previousLayerNeuronCount);
        }
    }

    public void computeActivation(double[] input) {
        for(int i = 0; i < activation.length; i++) {
            double signal = neurons[i].signal(input, weightedInput);
            activation[i] = activationFunction.f(signal);
            activation1[i] = activationFunction.f1(signal);
        }
    }

    public void backpropagate(double[] previousLayerActivation, double[] previousLayerActivation1) {
        for(int i = 0; i < neurons.length; i++) {
            Vector.multiplyAndAdd(neurons[i].weights, error[i], previousLayerError);
        }

        Vector.multiply(previousLayerError, previousLayerActivation1, previousLayerError);

        for(int i = 0; i < neurons.length; i++) {
            Vector.multiplyAndAdd(previousLayerActivation, error[i], neurons[i].weightError);
        }
    }

    public void update(double rate) {
        for(int i = 0; i < neurons.length; i++) {
            neurons[i].update(rate);
            neurons[i].bias -= rate * error[i];
        }
    }
}
