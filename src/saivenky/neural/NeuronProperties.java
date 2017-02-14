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

    public double affine(NeuronSet inputNeurons) {
        double sum = 0;
        for(int i = 0; i < inputNeurons.size(); i++) {
            sum += weights[i] * inputNeurons.get(i).getActivation();
        }

        return sum + bias;
    }

    public double affineForSelected(NeuronSet inputNeurons) {
        return inputNeurons.affine(weights, bias);
    }

    synchronized void update(double rate) {
        Vector.multiplyAndAdd(weightCostGradient, -rate, weights);
        bias -= rate * biasCostGradient;
        Vector.zero(weightCostGradient);
        biasCostGradient = 0;
    }

    void addError(NeuronSet inputNeurons, double cost) {
        biasCostGradient += cost;
        inputNeurons.addToWeightError(weightCostGradient, cost);
    }

    double[] getWeights() {
        return weights;
    }

    double getBias() {
        return bias;
    }
}
