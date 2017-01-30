package saivenky.neural;

import saivenky.neural.neuron.NeuronInitializer;

/**
 * Created by saivenky on 1/26/17.
 */
public class Neuron {
    public static double signal(double[] weight, double[] input, double[] weightedInput, double bias) {
        Vector.multiply(weight, input, weightedInput);
        return Vector.sum(weightedInput) + bias;
    }

    public static void initializeWeights(NeuronInitializer neuronInitializer, double[] weights) {
        for(int i = 0; i < weights.length; i++) {
            weights[i] = neuronInitializer.createWeight();
        }
    }

    double[] weights;
    double bias;
    double[] weightError;
    double biasError;

    public Neuron(NeuronInitializer neuronInitializer, int previousLayerNeurons) {
        weights = new double[previousLayerNeurons];
        weightError = new double[previousLayerNeurons];
        initializeWeights(neuronInitializer, weights);
        bias = neuronInitializer.createBias();
    }

    public double signal(double[] input, double[] weightedInput) {
        return signal(weights, input, weightedInput, bias);
    }

    public void update(double rate) {
        Vector.multiplyAndAdd(weightError, -rate, weights);
        bias -= rate * biasError;
        Vector.zero(weightError);
        biasError = 0;
    }

    @Override
    public String toString() {
        return String.format("bias: %.4f, weights: %s", bias, Vector.str(weights));
    }
}
