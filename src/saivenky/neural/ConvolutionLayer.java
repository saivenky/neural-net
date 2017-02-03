package saivenky.neural;

import saivenky.neural.activation.ActivationFunction;
import saivenky.neural.neuron.GaussianInitializer;

/**
 * Created by saivenky on 2/1/17.
 */
public class ConvolutionLayer extends Layer {
    Spatial3DStructure spatial3DStructure;
    private ActivationFunction activationFunction;

    public ConvolutionLayer(
            ActivationFunction activationFunction, int frames, int segmentWidth, int segmentHeight, Spatial2DStructure spatial2DStructure) {
        super(null);
        this.activationFunction = activationFunction;
        initializeNeurons(frames, segmentWidth, segmentHeight, spatial2DStructure);
        setDropoutRate(0);
    }

    private void initializeNeurons(int frames, int segmentWidth, int segmentHeight, Spatial2DStructure spatial2DStructure) {
        Neuron[] neuronArray = new Neuron[
                frames * (spatial2DStructure.width - segmentWidth + 1) * (spatial2DStructure.height - segmentHeight + 1)];
        spatial3DStructure = new Spatial3DStructure(
                neuronArray, spatial2DStructure.width - segmentWidth + 1, spatial2DStructure.height - segmentHeight + 1, frames);

        GaussianInitializer gaussianInitializer = new GaussianInitializer();
        NeuronProperties[] properties = new NeuronProperties[frames];
        for(int f = 0; f < frames; f++) {
            properties[f] = new NeuronProperties(gaussianInitializer, segmentWidth * segmentHeight);
        }
        for(int x = 0; x <= spatial2DStructure.width - segmentWidth; x++) {
            for(int y = 0; y <= spatial2DStructure.height - segmentHeight; y++) {
                Neuron[] segment = spatial2DStructure.getSegment(
                        x, x + segmentWidth - 1, y, y + segmentHeight - 1);
                NeuronSet inputNeuronSet = new NeuronSet(segment);
                for(int f = 0; f < frames; f++) {
                    spatial3DStructure.set(x, y, f, new Neuron(properties[f], inputNeuronSet, activationFunction));
                }
            }
        }

        neurons = new NeuronSet(neuronArray);
    }
}
