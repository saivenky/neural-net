package saivenky.neural;

/**
 * Created by saivenky on 2/1/17.
 */
public class MaxPoolingLayer extends Layer {
    Spatial3DStructure spatial3DStructure;

    public MaxPoolingLayer(int poolWidth, int poolHeight, ConvolutionLayer convolutionLayer) {
        super( null);
        initializeNeurons(poolWidth, poolHeight, convolutionLayer.spatial3DStructure);
    }

    void initializeNeurons(int poolWidth, int poolHeight, Spatial3DStructure input3DStructure) {
        int frames = input3DStructure.depth;
        INeuron[] neuronArray = new INeuron[
                frames * (input3DStructure.width / poolWidth) * (input3DStructure.height / poolHeight)];
        spatial3DStructure = new Spatial3DStructure(
                neuronArray, input3DStructure.width / poolWidth, input3DStructure.height / poolHeight, frames);

        for(int x = 0; x < input3DStructure.width / poolWidth; x++) {
            for(int y = 0; y < input3DStructure.height / poolHeight; y++) {
                for(int z = 0; z < frames; z++) {
                    INeuron[] segment = input3DStructure.getSegmentSlice(
                            x * poolWidth, (x + 1) * poolWidth - 1, y * poolHeight, (y + 1) * poolHeight - 1, z);
                    NeuronSet neuronSet = new NeuronSet(segment);
                    spatial3DStructure.set(x, y, z, new MaxPoolingNeuron(neuronSet));
                }
            }
        }

        neurons = new NeuronSet(neuronArray);
    }
}
