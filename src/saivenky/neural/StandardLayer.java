package saivenky.neural;

import saivenky.neural.activation.ActivationFunction;
import saivenky.neural.neuron.NeuronInitializer;

/**
 * Created by saivenky on 1/31/17.
 */
public class StandardLayer extends Layer {
    public StandardLayer(
            int neuronCount, NeuronSet previousLayerNeurons, ActivationFunction activationFunction, NeuronInitializer neuronInitializer) {
        super(new NeuronSet(new Neuron[neuronCount]));
        initializeNeurons(neuronInitializer, previousLayerNeurons, activationFunction);
        setDropoutRate(0);
    }

    private void initializeNeurons(
            NeuronInitializer neuronInitializer, NeuronSet previousLayerNeurons, ActivationFunction activationFunction) {
        for(int i = 0; i < neurons.size(); i++) {
            neurons.set(i, new Neuron(neuronInitializer, previousLayerNeurons, activationFunction));
        }
    }
}
