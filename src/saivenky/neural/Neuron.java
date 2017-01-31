package saivenky.neural;

import saivenky.neural.neuron.NeuronInitializer;
import saivenky.neural.neuron.NeuronProperties;

/**
 * Created by saivenky on 1/26/17.
 */
public class Neuron {
    final NeuronProperties properties;

    Neuron(NeuronInitializer neuronInitializer, int previousLayerNeurons) {
        properties = new NeuronProperties(neuronInitializer, previousLayerNeurons);
    }

    double signal(double[] input, double[] weightedInput) {
        return properties.affine(input, weightedInput);
    }

    double signalForSelected(double[] input, double[] weightedInput, int[] selected) {
        return properties.affineForSelected(input, weightedInput, selected);
    }

    double signalScaled(double[] input, double[] weightedInput, double scale) {
        return properties.affineScaled(input, weightedInput, scale);
    }

    void update(double rate) {
        properties.update(rate);
    }
}
