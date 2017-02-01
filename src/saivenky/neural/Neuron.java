package saivenky.neural;

import saivenky.neural.neuron.NeuronInitializer;
import saivenky.neural.neuron.NeuronProperties;

/**
 * Created by saivenky on 1/26/17.
 */
public class Neuron {
    final NeuronProperties properties;
    final NeuronSet inputNeurons;

    Neuron(NeuronInitializer neuronInitializer, NeuronSet inputNeurons, int previousLayerNeurons) {
        this.inputNeurons = inputNeurons;
        properties = new NeuronProperties(neuronInitializer, previousLayerNeurons);
    }

    double signalForSelected() {
        return properties.affineForSelected(inputNeurons.activation, inputNeurons.selected);
    }

    double signalScaled(double scale) {
        return properties.affineScaled(inputNeurons.activation, scale);
    }

    void update(double rate) {
        properties.update(rate);
    }
}
