package saivenky.neural;

import saivenky.neural.neuron.NeuronInitializer;

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

    public NeuronProperties(NeuronInitializer neuronInitializer, int inputSize) {
        this(inputSize);
        initializeWeights(neuronInitializer, weights);
        bias = neuronInitializer.createBias();
    }

    private NeuronProperties(int inputSize) {
        weights = new double[inputSize];
        weightCostGradient = new double[inputSize];
    }

    public double affine(NeuronSet input) {
        return affine(weights, input, bias);
    }

    public double affineForSelected(NeuronSet input) {
        return affineForSelected(weights, input, bias);
    }

    public void update(double rate) {
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
