package saivenky.neural.c;

import saivenky.neural.IOutputLayer;

import java.nio.ByteBuffer;

/**
 * Created by saivenky on 2/20/17.
 */
public class OutputLayer extends Layer implements IOutputLayer {
    private final int size;

    public OutputLayer(Layer previousLayer, int miniBatchSize) {
        shape = previousLayer.shape;
        size = previousLayer.shape[0] * previousLayer.shape[1] * previousLayer.shape[2];
        outputSignals = new ByteBuffer[miniBatchSize];
        outputErrors = new ByteBuffer[miniBatchSize];
        nativeLayerPtr = create(size, previousLayer.nativeLayerPtr);
        adjustByteOrderOnBuffers();
    }

    private native long create(int size, long previousLayerNativePtr);
    private native long destroy(long nativeLayerPtr);

    @Override
    public void feedforward() {
    }

    @Override
    public void backpropagate() {
    }

    @Override
    public void gradientDescent(float rate) {
    }

    @Override
    public void setExpected(float[][] expected) {
    }

    @Override
    public void setSignalCostGradient(float[][] cost) {
        for (int b = 0; b < cost.length; b++) {
            for (int i = 0, bbIndex = 0; i < cost.length; i++, bbIndex += SIZEOF_FLOAT_T) {
                outputErrors[b].putFloat(bbIndex, cost[b][i]);
            }
        }
    }

    @Override
    public void getPredicted(float[][] predicted) {
        for(int b = 0; b < predicted.length; b++) {
            for(int i = 0, bbIndex = 0; i < size; i++, bbIndex += SIZEOF_FLOAT_T) {
                predicted[b][i] = outputSignals[b].getFloat(bbIndex);
            }
        }
    }

    @Override
    public int size() {
        return size;
    }
}
