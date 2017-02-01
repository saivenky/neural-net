package saivenky.neural;

import saivenky.neural.activation.ActivationFunction;
import saivenky.neural.neuron.NeuronInitializer;

/**
 * Created by saivenky on 1/31/17.
 */
public class StandardLayer extends Layer {
    StandardLayer(
            int neuronCount, NeuronSet previousLayerNeurons, int previousLayerNeuronCount, ActivationFunction activationFunction, NeuronInitializer neuronInitializer) {
        super(activationFunction);
        neurons = new NeuronSet(new Neuron[neuronCount]);
        initializeNeurons(neuronInitializer, previousLayerNeurons, previousLayerNeuronCount);
        setDropoutRate(0);
    }

    private void initializeNeurons(
            NeuronInitializer neuronInitializer, NeuronSet previousLayerNeurons, int previousLayerNeuronCount) {
        for(int i = 0; i < neurons.size(); i++) {
            neurons.set(i, new Neuron(neuronInitializer, previousLayerNeurons, previousLayerNeuronCount));
        }
    }
}
