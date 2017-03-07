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
    public void gradientDescent(double rate) {
    }

    @Override
    public void setExpected(double[][] expected) {
    }

    @Override
    public void setSignalCostGradient(double[][] cost) {
        for (int b = 0; b < cost.length; b++) {
            for (int i = 0, bbIndex = 0; i < cost.length; i++, bbIndex += SIZEOF_DOUBLE) {
                outputErrors[b].putDouble(bbIndex, cost[b][i]);
            }
        }
    }

    @Override
    public void getPredicted(double[][] predicted) {
        for(int b = 0; b < predicted.length; b++) {
            for(int i = 0, bbIndex = 0; i < size; i++, bbIndex += SIZEOF_DOUBLE) {
                predicted[b][i] = outputSignals[b].getDouble(bbIndex);
            }
        }
    }

    @Override
    public int size() {
        return size;
    }
}
