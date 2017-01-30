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
    private int[] inputNonDropout;

    double dropoutRate;
    int[] nonDropout;

    Layer(
            int neuronCount, int previousLayerNeuronCount, ActivationFunction activationFunction, NeuronInitializer neuronInitializer) {
        weightedInput = new double[previousLayerNeuronCount];
        neurons = new Neuron[neuronCount];
        initializeNeurons(neuronInitializer, previousLayerNeuronCount);
        activation = new double[neuronCount];
        activation1 = new double[neuronCount];
        this.activationFunction = activationFunction;
        previousLayerError = new double[previousLayerNeuronCount];
        setDropoutRate(0);

    }

    public void setDropoutRate(double dropoutRate) {
        this.dropoutRate = dropoutRate;
        int nonDropoutLength = neurons.length - (int)(neurons.length * dropoutRate);
        nonDropout = new int[nonDropoutLength];
        Vector.select(nonDropout, neurons.length);
    }

    void reselectDropout() {
        Vector.select(nonDropout, neurons.length);
    }

    private void initializeNeurons(
            NeuronInitializer neuronInitializer, int previousLayerNeuronCount) {
        for(int i = 0; i < neurons.length; i++) {
            neurons[i] = new Neuron(neuronInitializer, previousLayerNeuronCount);
        }
    }

    private void run(double[] input, double[] input1) {
        this.input = input;
        this.input1 = input1;
        this.inputNonDropout = null;

        for(int i : nonDropout) {
            double signal = neurons[i].signal(input, weightedInput);
            activation[i] = activationFunction.f(signal);
            activation1[i] = activationFunction.f1(signal);
        }
    }

    void run(double[] input, double[] input1, int[] inputNonDropout) {
        if (inputNonDropout == null) {
            run(input, input1);
            return;
        }

        this.input = input;
        this.input1 = input1;
        this.inputNonDropout = inputNonDropout;

        for(int i : nonDropout) {
            double signal = neurons[i].signalForSelected(input, weightedInput, inputNonDropout);
            activation[i] = activationFunction.f(signal);
            activation1[i] = activationFunction.f1(signal);
        }
    }

    void runScaled(double[] input, double[] input1, double inputDropoutRate) {
        this.input = input;
        this.input1 = input1;
        double scale = signalScaleFromDropout(inputDropoutRate);

        for(int i = 0; i < neurons.length; i++) {
            double signal = neurons[i].signalScaled(input, weightedInput, scale);
            activation[i] = activationFunction.f(signal);
            activation1[i] = activationFunction.f1(signal);
        }
    }

    void backpropagate() {
        if(input1 != null) {
            for(int i : nonDropout) {
                Vector.multiplyAndAddSelected(neurons[i].weights, error[i], previousLayerError, inputNonDropout);
            }

            Vector.multiplySelected(previousLayerError, input1, previousLayerError, inputNonDropout);
        }

        for(int i : nonDropout) {
            neurons[i].biasError += error[i];
            Vector.multiplyAndAddSelected(input, error[i], neurons[i].weightError, inputNonDropout);
        }
    }

    void update(double rate) {
        for(int i : nonDropout) {
            Neuron neuron = neurons[i];
            neuron.update(rate);
        }
        Vector.zero(previousLayerError);
    }

    private static double signalScaleFromDropout(double dropoutRate) {
        return (1 - dropoutRate);
    }
}
