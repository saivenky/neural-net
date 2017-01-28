package saivenky.neural;

/**
 * Created by saivenky on 1/26/17.
 */
public class Neuron {
    public static double signal(double[] weight, double[] input, double[] weightedInput, double bias) {
        Vector.multiply(weight, input, weightedInput);
        return Vector.sum(weightedInput) + bias;
    }

    double[] weights;
    double bias;
    double[] weightError;
    double biasError;

    public Neuron(int previousLayerNeurons) {
        weights = new double[previousLayerNeurons];
        weightError = new double[previousLayerNeurons];
        Vector.random(weights);
        bias = 0.1;
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
