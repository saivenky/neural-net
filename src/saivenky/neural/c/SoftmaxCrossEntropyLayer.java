package saivenky.neural.c;

import saivenky.neural.IOutputLayer;
import saivenky.neural.NeuronSet;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.nio.ByteBuffer;

/**
 * Created by saivenky on 2/25/17.
 */
public class SoftmaxCrossEntropyLayer extends Layer implements IOutputLayer {
    private final int size;

    public SoftmaxCrossEntropyLayer(Layer previousLayer) {
        shape = previousLayer.shape;
        size = previousLayer.shape[0] * previousLayer.shape[1] * previousLayer.shape[2];
        nativeLayerPtr = create(size, previousLayer.outputSignal, previousLayer.outputError);
        adjustByteOrderOnBuffers();
        outputError = inputError;
    }

    private native long create(int size, ByteBuffer inputActivation, ByteBuffer inputError);
    private native long destroy(long nativeLayerPtr);
    private native void feedforward(long nativeLayerPtr);
    private native void setExpected(long nativeLayerPtr, double[] expected);

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
    }

    @Override
    public void gradientDescent(double rate) {
    }

    @Override
    public void setSignalCostGradient(double[] cost) {
    }

    public void setExpected(double[] expected) {
        setExpected(nativeLayerPtr, expected);
    }

    @Override
    public void getPredicted(double[] predicted) {
        for(int i = 0, bbIndex = 0; i < size; i++, bbIndex += SIZEOF_DOUBLE) {
            predicted[i] = outputSignal.getDouble(bbIndex);
        }
    }

    @Override
    public int size() {
        return size;
    }
}
