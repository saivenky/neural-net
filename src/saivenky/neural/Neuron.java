package saivenky.neural;

import java.util.Random;

/**
 * Created by saivenky on 1/26/17.
 */
public class Neuron {
    public static double signal(double[] weight, double[] input, double[] weightedInput, double bias) {
        Vector.multiply(weight, input, weightedInput);
        return Vector.sum(weightedInput) + bias;
    }

    private static final Random random = new Random(1);
    private static int neurons = 0;

    double[] weights;
    double bias;
    double[] weightError;
    public Neuron(int previousLayerNeurons) {
        weights = new double[previousLayerNeurons];
        weightError = new double[previousLayerNeurons];
        Vector.random(weights);
        bias = random.nextGaussian();
    }

    public double signal(double[] input, double[] weightedInput) {
        return signal(weights, input, weightedInput, bias);
    }

    public void update(double learningRate) {
        Vector.multiplyAndAdd(weightError, -learningRate, weights);
        Vector.zero(weightError);
    }

    @Override
    public String toString() {
        return String.format("bias: %.4f, weights: %s", bias, Vector.str(weights));
    }
}
