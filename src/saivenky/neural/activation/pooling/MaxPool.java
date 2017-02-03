package saivenky.neural.activation.pooling;

import saivenky.neural.Neuron;

/**
 * Created by saivenky on 2/1/17.
 */
public class MaxPool implements PoolingActivationFunction {
    private static PoolingActivationFunction SINGLETON;

    public static PoolingActivationFunction getInstance() {
        if (SINGLETON == null) {
            SINGLETON = new MaxPool();
        }

        return SINGLETON;
    }

    private MaxPool() {}

    @Override
    public double pool(Neuron[] neurons, double[] activation1) {
        int maxIndex = -1;
        double max = -1;
        for (int i = 0; i < neurons.length; i++) {
            activation1[i] = 0;
            if(neurons[i].activation > max) {
                maxIndex = i;
                max = neurons[i].activation;
            }
        }

        activation1[maxIndex] = 1;
        return max;
    }
}
