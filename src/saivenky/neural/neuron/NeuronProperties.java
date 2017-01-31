package saivenky.neural.neuron;

import saivenky.neural.Vector;

/**
 * Created by saivenky on 1/30/17.
 */
public class NeuronProperties {
    public double[] weights;
    public double bias;
    public double[] weightCostGradient;
    public double biasCostGradient;

    private static void initializeWeights(NeuronInitializer neuronInitializer, double[] weights) {
        for(int i = 0; i < weights.length; i++) {
            weights[i] = neuronInitializer.createWeight();
        }
    }

    public NeuronProperties(NeuronInitializer neuronInitializer, int previousLayerNeurons) {
        weights = new double[previousLayerNeurons];
        weightCostGradient = new double[previousLayerNeurons];
        initializeWeights(neuronInitializer, weights);
        bias = neuronInitializer.createBias();
    }

    public double affine(double[] input, double[] weightedInput) {
        return affine(weights, input, weightedInput, bias);
    }

    public double affineScaled(double[] input, double[] weightedInput, double scale) {
        return affineScaled(weights, input, weightedInput, bias, scale);
    }

    public double affineForSelected(double[] input, double[] weightedInput, int[] selected) {
        return affineForSelected(weights, input, weightedInput, bias, selected);
    }

    public void update(double rate) {
        Vector.multiplyAndAdd(weightCostGradient, -rate, weights);
        bias -= rate * biasCostGradient;
        Vector.zero(weightCostGradient);
        biasCostGradient = 0;
    }

    private static double affine(double[] weight, double[] input, double[] weightedInput, double bias) {
        Vector.multiply(weight, input, weightedInput);
        return Vector.sum(weightedInput) + bias;
    }

    private static double affineScaled(double[] weight, double[] input, double[] weightedInput, double bias, double scale) {
        Vector.multiply(weight, input, weightedInput);
        return scale * Vector.sum(weightedInput) + bias;
    }

    private static double affineForSelected(double[] weight, double[] input, double[] weightedInput, double bias, int[] selected) {
        Vector.multiplySelected(weight, input, weightedInput, selected);
        return Vector.sumSelected(weightedInput, selected) + bias;
    }
}
