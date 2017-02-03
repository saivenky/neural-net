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

    public NeuronProperties(NeuronInitializer neuronInitializer, int previousLayerNeurons) {
        weights = new double[previousLayerNeurons];
        weightCostGradient = new double[previousLayerNeurons];
        initializeWeights(neuronInitializer, weights);
        bias = neuronInitializer.createBias();
    }

    public double affineScaled(NeuronSet input, double scale) {
        return affineScaled(weights, input, bias, scale);
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

    private static double affineScaled(double[] weight, NeuronSet input, double bias, double scale) {
        //Vector.multiply(weight, input, weightedInput);
        double sum = 0;
        for(int i = 0; i < weight.length; i++) {
            sum += weight[i] * input.get(i).activation;
        }
        return scale * sum + bias;
    }

    private static double affineForSelected(double[] weight, NeuronSet input, double bias) {
        //Vector.multiplySelected(weight, input, weightedInput, selected);
        if(input.selected == null) {
            return affineScaled(weight, input, bias, 1);
        }

        double sum = 0;
        for(int i : input.selected) {
            sum += weight[i] * input.get(i).activation;
        }

        return sum + bias;
    }
}
