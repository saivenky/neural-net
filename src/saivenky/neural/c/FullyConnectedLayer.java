package saivenky.neural.c;

import saivenky.neural.NeuronSet;

/**
 * Created by saivenky on 2/22/17.
 */
public class FullyConnectedLayer extends Layer {
    public FullyConnectedLayer(Layer previousLayer, int outputSize) {
        int inputSize = previousLayer.shape[0] * previousLayer.shape[1] * previousLayer.shape[2];
        shape = new int[] {
                outputSize,
                1,
                1
        };

        nativeLayerPtr = create(inputSize, outputSize, previousLayer.nativeLayerPtr);
        adjustByteOrderOnBuffers();
    }

    private native long create(long inputSize, long outputSize, long previousLayerNativePtr);
    private native long destroy(long nativeLayerPtr);

    @Override
    public NeuronSet getNeurons() {
        return null;
    }

    @Override
    public void run() {
        feedforward();
    }

    @Override
    public void feedforward() {
        feedforward(nativeLayerPtr);
    }

    @Override
    public void backpropagate(boolean backpropagateToPreviousLayer) {
        backpropogate(nativeLayerPtr);
    }

    @Override
    public void gradientDescent(double rate) {
        update(nativeLayerPtr, rate);
    }
}
