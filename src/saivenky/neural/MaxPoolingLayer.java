package saivenky.neural;

/**
 * Created by saivenky on 2/1/17.
 */
public class MaxPoolingLayer extends Layer {
    public MaxPoolingLayer(int poolWidth, int poolHeight, ILayer previousLayer) {
        super( null);
        Spatial3DStructure input3DStructure = previousLayer.getNeurons();
        initializeNeurons(poolWidth, poolHeight, input3DStructure);
    }

    private void initializeNeurons(int poolWidth, int poolHeight, Spatial3DStructure input3DStructure) {
        int outputWidth = FilterDimensionCalculator.calculateOutputSize(input3DStructure.getWidth(), poolWidth, poolWidth, 0);
        int outputHeight = FilterDimensionCalculator.calculateOutputSize(input3DStructure.getHeight(), poolHeight, poolHeight, 0);
        int outputDepth = input3DStructure.getDepth();

        INeuron[] neuronArray = new INeuron[outputWidth * outputHeight * outputDepth];
        neurons = new NeuronSet(neuronArray);
        neurons.setShape(outputWidth, outputHeight, outputDepth);

        for(int x = 0, inX = 0; x < neurons.getWidth(); x++, inX+=poolWidth) {
            for(int y = 0, inY = 0; y < neurons.getHeight(); y++, inY+=poolHeight) {
                for(int z = 0; z < neurons.getDepth(); z++) {
                    INeuron[] segment = input3DStructure.getSegment(inX, inY, z, poolWidth, poolHeight, 1);
                    NeuronSet neuronSet = new NeuronSet(segment);
                    neurons.set(x, y, z, new MaxPoolingNeuron(neuronSet));
                }
            }
        }
    }
}
