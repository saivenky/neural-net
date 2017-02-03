package saivenky.neural.activation.pooling;

import saivenky.neural.Neuron;

/**
 * Created by saivenky on 2/1/17.
 */
public interface PoolingActivationFunction {
    double pool(Neuron[] neurons, double[] weights);
}
