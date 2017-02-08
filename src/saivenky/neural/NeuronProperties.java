package saivenky.neural;

import saivenky.neural.neuron.NeuronInitializer;

/**
 * Created by saivenky on 1/30/17.
 */
class NeuronProperties {
    double[] weights;
    double bias;
    double[] weightCostGradient;
    double biasCostGradient;

    private static void initializeWeights(NeuronInitializer neuronInitializer, double[] weights) {
        for(int i = 0; i < weights.length; i++) {
            weights[i] = neuronInitializer.createWeight();
        }
    }

    NeuronProperties(NeuronInitializer neuronInitializer, int inputSize) {
        this(inputSize);
        initializeWeights(neuronInitializer, weights);
        bias = neuronInitializer.createBias();
    }

    private NeuronProperties(int inputSize) {
        weights = new double[inputSize];
        weightCostGradient = new double[inputSize];
    }

    double affine(NeuronSet input) {
        return affine(weights, input, bias);
    }

    double affineForSelected(NeuronSet input) {
        return affineForSelected(weights, input, bias);
    }

    synchronized void update(double rate) {
        Vector.multiplyAndAdd(weightCostGradient, -rate, weights);
        bias -= rate * biasCostGradient;
        Vector.zero(weightCostGradient);
        biasCostGradient = 0;
    }

    private static double affine(double[] weight, NeuronSet input, double bias) {
        double sum = 0;
        for(int i = 0; i < weight.length; i++) {
            sum += weight[i] * input.get(i).getActivation();
        }
        return sum + bias;
    }

    private static double affineForSelected(double[] weight, NeuronSet input, double bias) {
        if(input.selected == null) {
            return affine(weight, input, bias);
        }

        double sum = 0;
        for(int i : input.selected) {
            sum += weight[i] * input.get(i).getActivation();
        }

        return sum + bias;
    }
}
