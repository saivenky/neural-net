package saivenky.neural;

import saivenky.neural.activation.ActivationFunction;
import saivenky.neural.neuron.NeuronInitializer;

/**
 * Created by saivenky on 2/1/17.
 */
public class ConvolutionLayer extends Layer {

    final NeuronProperties[] properties;

    public ConvolutionLayer(
            int frames, int segmentWidth, int segmentHeight, ILayer previousLayer, ActivationFunction activationFunction, NeuronInitializer neuronInitializer) {
        super(null);
        Spatial3DStructure input3DStructure = previousLayer.getNeurons();
        properties = new NeuronProperties[frames];
        for(int f = 0; f < frames; f++) {
            properties[f] = new NeuronProperties(neuronInitializer, segmentWidth * segmentHeight * input3DStructure.getDepth());
        }

        initializeNeurons(properties, frames, segmentWidth, segmentHeight, input3DStructure, activationFunction);
    }

    private void initializeNeurons(NeuronProperties[] properties, int frames, int segmentWidth, int segmentHeight,
                                   Spatial3DStructure input3DStructure,
                                   ActivationFunction activationFunction) {
        int outputWidth = FilterDimensionCalculator.calculateOutputSize(input3DStructure.getWidth(), segmentWidth, 1);
        int outputHeight = FilterDimensionCalculator.calculateOutputSize(input3DStructure.getHeight(), segmentHeight, 1);
        Neuron[] neuronArray = new Neuron[frames * outputWidth * outputHeight];
        neurons = new NeuronSet(neuronArray);
        neurons.setShape(outputWidth, outputHeight, frames);

        for(int x = 0; x < neurons.getWidth(); x++) {
            for(int y = 0; y < neurons.getHeight(); y++) {
                INeuron[] segment = input3DStructure.getSegment(
                        x, y, 0, segmentWidth, segmentHeight, input3DStructure.getDepth());
                NeuronSet inputNeuronSet = new NeuronSet(segment);
                for(int f = 0; f < frames; f++) {
                    neurons.set(x, y, f, new Neuron(properties[f], inputNeuronSet, activationFunction));
                }
            }
        }
    }
}
