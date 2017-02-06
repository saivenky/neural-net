package saivenky.neural;

/**
 * Created by saivenky on 2/1/17.
 */
public class MaxPoolingLayer extends Layer {
    public MaxPoolingLayer(int poolWidth, int poolHeight, Spatial3DStructure input3DStructure) {
        super( null);
        initializeNeurons(poolWidth, poolHeight, input3DStructure);
    }

    void initializeNeurons(int poolWidth, int poolHeight, Spatial3DStructure input3DStructure) {
        int frames = input3DStructure.getDepth();
        INeuron[] neuronArray = new INeuron[
                frames * (input3DStructure.getWidth() / poolWidth) * (input3DStructure.getHeight() / poolHeight)];
        neurons = new ThreadedNeuronSet(neuronArray);
        neurons.setShape(input3DStructure.getWidth() / poolWidth, input3DStructure.getHeight() / poolHeight, frames);

        for(int x = 0; x < input3DStructure.getWidth() / poolWidth; x++) {
            for(int y = 0; y < input3DStructure.getHeight() / poolHeight; y++) {
                for(int z = 0; z < frames; z++) {
                    INeuron[] segment = input3DStructure.getSegmentSlice(
                            x * poolWidth, (x + 1) * poolWidth - 1, y * poolHeight, (y + 1) * poolHeight - 1, z);
                    NeuronSet neuronSet = new ThreadedNeuronSet(segment);
                    neurons.set(x, y, z, new MaxPoolingNeuron(neuronSet));
                }
            }
        }
    }
}
