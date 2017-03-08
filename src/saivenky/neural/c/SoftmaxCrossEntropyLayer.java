package saivenky.neural.c;

import saivenky.neural.IOutputLayer;

import java.nio.ByteBuffer;

/**
 * Created by saivenky on 2/25/17.
 */
public class SoftmaxCrossEntropyLayer extends Layer implements IOutputLayer {
    private final int size;

    public SoftmaxCrossEntropyLayer(Layer previousLayer, int miniBatchSize) {
        shape = previousLayer.shape;
        size = previousLayer.shape[0] * previousLayer.shape[1] * previousLayer.shape[2];
        outputSignals = new ByteBuffer[miniBatchSize];
        outputErrors = new ByteBuffer[miniBatchSize];
        nativeLayerPtr = create(size, previousLayer.nativeLayerPtr);
        adjustByteOrderOnBuffers();
    }

    private native long create(int size, long previousLayerNativePtr);
    private native long destroy(long nativeLayerPtr);
    private native void setExpected(long nativeLayerPtr, float[][] expected);

    @Override
    public void gradientDescent(float rate) {
    }

    @Override
    public void setSignalCostGradient(float[][] cost) {
    }

    public void setExpected(float[][] expected) {
        setExpected(nativeLayerPtr, expected);
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
