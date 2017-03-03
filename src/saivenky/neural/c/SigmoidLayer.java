package saivenky.neural.c;

import saivenky.neural.NeuronSet;

/**
 * Created by saivenky on 2/19/17.
 */
public class SigmoidLayer extends Layer {
    public SigmoidLayer(Layer previousLayer) {
        shape = previousLayer.shape;
        int size = previousLayer.shape[0] * previousLayer.shape[1] * previousLayer.shape[2];

        nativeLayerPtr = create(size, previousLayer.nativeLayerPtr);
        adjustByteOrderOnBuffers();
    }

    private native long create(int size, long previousLayerNativePtr);
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
    }
}
