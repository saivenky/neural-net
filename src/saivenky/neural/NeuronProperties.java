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
        double sum = 0;
        for(int i = 0; i < weights.length; i++) {
            sum += weights[i] * input.get(i).getActivation();
        }

        return sum + bias;
    }

    double affineForSelected(NeuronSet input) {
        double sum = 0;
        for(int i : input.selected) {
            sum += weights[i] * input.get(i).getActivation();
        }

        return sum + bias;
    }

    synchronized void update(double rate) {
        Vector.multiplyAndAdd(weightCostGradient, -rate, weights);
        bias -= rate * biasCostGradient;
        Vector.zero(weightCostGradient);
        biasCostGradient = 0;
    }

    void addError(NeuronSet inputNeurons, double cost) {
        biasCostGradient += cost;
        for(int i : inputNeurons.selected) {
            weightCostGradient[i] += inputNeurons.get(i).getActivation() * cost;
        }
    }

    double[] getWeights() {
        return weights;
    }

    double getBias() {
        return bias;
    }
}
