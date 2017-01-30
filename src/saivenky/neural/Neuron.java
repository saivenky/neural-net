package saivenky.neural;

import saivenky.neural.neuron.NeuronInitializer;

/**
 * Created by saivenky on 1/26/17.
 */
public class Neuron {
    private static double signal(double[] weight, double[] input, double[] weightedInput, double bias) {
        Vector.multiply(weight, input, weightedInput);
        return Vector.sum(weightedInput) + bias;
    }

    private static double signalForSelected(double[] weight, double[] input, double[] weightedInput, double bias, int[] selected) {
        Vector.multiplySelected(weight, input, weightedInput, selected);
        return Vector.sumSelected(weightedInput, selected) + bias;
    }

    private static double signalScaled(double[] weight, double[] input, double[] weightedInput, double bias, double scale) {
        Vector.multiply(weight, input, weightedInput);
        return scale * Vector.sum(weightedInput) + bias;
    }

    private static void initializeWeights(NeuronInitializer neuronInitializer, double[] weights) {
        for(int i = 0; i < weights.length; i++) {
            weights[i] = neuronInitializer.createWeight();
        }
    }

    double[] weights;
    double bias;
    double[] weightError;
    double biasError;

    Neuron(NeuronInitializer neuronInitializer, int previousLayerNeurons) {
        weights = new double[previousLayerNeurons];
        weightError = new double[previousLayerNeurons];
        initializeWeights(neuronInitializer, weights);
        bias = neuronInitializer.createBias();
    }

    double signal(double[] input, double[] weightedInput) {
        return signal(weights, input, weightedInput, bias);
    }

    double signalForSelected(double[] input, double[] weightedInput, int[] selected) {
        return signalForSelected(weights, input, weightedInput, bias, selected);
    }

    double signalScaled(double[] input, double[] weightedInput, double scale) {
        return signalScaled(weights, input, weightedInput, bias, scale);
    }

    void update(double rate) {
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
