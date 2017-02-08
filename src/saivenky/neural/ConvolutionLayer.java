package saivenky.neural;

import saivenky.neural.activation.ActivationFunction;
import saivenky.neural.neuron.GaussianInitializer;

/**
 * Created by saivenky on 2/1/17.
 */
public class ConvolutionLayer extends Layer {
    private ActivationFunction activationFunction;

    public ConvolutionLayer(
            ActivationFunction activationFunction, int frames, int segmentWidth, int segmentHeight, Spatial3DStructure input3DStructure) {
        super(null);
        this.activationFunction = activationFunction;

        NeuronProperties[] properties = new NeuronProperties[frames];
        for(int f = 0; f < frames; f++) {
            properties[f] = new NeuronProperties(GaussianInitializer.getInstance(), segmentWidth * segmentHeight * input3DStructure.getDepth());
        }

        initializeNeurons(properties, frames, segmentWidth, segmentHeight, input3DStructure);
    }

    private void initializeNeurons(NeuronProperties[] properties, int frames, int segmentWidth, int segmentHeight, Spatial3DStructure input3DStructure) {
        int outputWidth = FilterDimensionCalculator.calculateOutputSize(input3DStructure.getWidth(), segmentWidth, 1);
        int outputHeight = FilterDimensionCalculator.calculateOutputSize(input3DStructure.getHeight(), segmentHeight, 1);
        Neuron[] neuronArray = new Neuron[frames * outputWidth * outputHeight];
        neurons = new ThreadedNeuronSet(neuronArray);
        neurons.setShape(outputWidth, outputHeight, frames);

        for(int x = 0; x < neurons.getWidth(); x++) {
            for(int y = 0; y < neurons.getHeight(); y++) {
                INeuron[] segment = input3DStructure.getSegment(
                        x, y, 0, segmentWidth, segmentHeight, input3DStructure.getDepth());
                NeuronSet inputNeuronSet = new ThreadedNeuronSet(segment);
                for(int f = 0; f < frames; f++) {
                    neurons.set(x, y, f, new Neuron(properties[f], inputNeuronSet, activationFunction));
                }
            }
        }
    }
}
