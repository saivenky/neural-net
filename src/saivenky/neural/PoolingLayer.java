package saivenky.neural;

import saivenky.neural.activation.pooling.PoolingActivationFunction;

/**
 * Created by saivenky on 2/1/17.
 */
public class PoolingLayer extends Layer {
    private PoolingActivationFunction poolingActivationFunction;
    Spatial3DStructure spatial3DStructure;

    public PoolingLayer(PoolingActivationFunction poolingActivationFunction, int poolWidth, int poolHeight, ConvolutionLayer convolutionLayer) {
        super( convolutionLayer.neurons);
        this.poolingActivationFunction = poolingActivationFunction;
        initializeNeurons(poolWidth, poolHeight, convolutionLayer.spatial3DStructure);
        setDropoutRate(0);
    }

    void initializeNeurons(int poolWidth, int poolHeight, Spatial3DStructure input3DStructure) {
        int frames = input3DStructure.depth;
        Neuron[] neuronArray = new Neuron[
                frames * (input3DStructure.width / poolWidth) * (input3DStructure.height / poolHeight)];
        spatial3DStructure = new Spatial3DStructure(
                neuronArray, input3DStructure.width / poolWidth, input3DStructure.height / poolHeight, frames);

        for(int x = 0; x < input3DStructure.width / poolWidth; x++) {
            for(int y = 0; y < input3DStructure.height / poolHeight; y++) {
                for(int z = 0; z < frames; z++) {
                    Neuron[] segment = input3DStructure.getSegmentSlice(
                            x * poolWidth, (x + 1) * poolWidth - 1, y * poolHeight, (y + 1) * poolHeight - 1, z);
                    NeuronSet neuronSet = new NeuronSet(segment);
                    spatial3DStructure.set(x, y, z, new PoolingNeuron(poolingActivationFunction, neuronSet));
                }
            }
        }

        neurons = new NeuronSet(neuronArray);
    }
}
