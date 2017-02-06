package saivenky.neural;

import saivenky.neural.activation.ActivationFunction;
import saivenky.neural.neuron.GaussianInitializer;

/**
 * Created by saivenky on 2/1/17.
 */
public class ConvolutionLayer extends Layer {
    private ActivationFunction activationFunction;

    public ConvolutionLayer(
            ActivationFunction activationFunction, int frames, int segmentWidth, int segmentHeight, Spatial2DStructure input2DStructure) {
        super(null);
        this.activationFunction = activationFunction;

        NeuronProperties[] properties = new NeuronProperties[frames];
        for(int f = 0; f < frames; f++) {
            properties[f] = new NeuronProperties(GaussianInitializer.getInstance(), segmentWidth * segmentHeight);
        }

        initializeNeurons(properties, frames, segmentWidth, segmentHeight, input2DStructure);
    }

    private void initializeNeurons(NeuronProperties[] properties, int frames, int segmentWidth, int segmentHeight, Spatial2DStructure spatial2DStructure) {
        Neuron[] neuronArray = new Neuron[
                frames * (spatial2DStructure.getWidth() - segmentWidth + 1) * (spatial2DStructure.getHeight() - segmentHeight + 1)];
        neurons = new ThreadedNeuronSet(neuronArray);
        neurons.setShape(spatial2DStructure.getWidth() - segmentWidth + 1, spatial2DStructure.getHeight() - segmentHeight + 1, frames);

        for(int x = 0; x < neurons.getWidth(); x++) {
            for(int y = 0; y < neurons.getHeight(); y++) {
                INeuron[] segment = spatial2DStructure.getSegment(
                        x, x + segmentWidth - 1, y, y + segmentHeight - 1);
                NeuronSet inputNeuronSet = new ThreadedNeuronSet(segment);
                for(int f = 0; f < frames; f++) {
                    neurons.set(x, y, f, new Neuron(properties[f], inputNeuronSet, activationFunction));
                }
            }
        }
    }
}
