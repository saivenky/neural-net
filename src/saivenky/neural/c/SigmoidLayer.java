package saivenky.neural.c;

import saivenky.neural.NeuronSet;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.nio.ByteBuffer;

/**
 * Created by saivenky on 2/19/17.
 */
public class SigmoidLayer extends Layer {
    public SigmoidLayer(Layer previousLayer) {
        inputActivation = previousLayer.outputSignal;
        inputError = previousLayer.outputError;
        shape = previousLayer.shape;
        int size = previousLayer.shape[0] * previousLayer.shape[1] * previousLayer.shape[2];

        nativeLayerPtr = create(size, inputActivation, inputError);
        adjustByteOrderOnBuffers();
    }

    private native long create(int size, ByteBuffer inputActivation, ByteBuffer inputError);
    private native long destroy(long nativeLayerPtr);
    private native void feedforward(long nativeLayerPtr);
    private native void backpropogate(long nativeLayerPtr);
    private native void update(long nativeLayerPtr, double rate);

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

    @Override
    public void setSignalCostGradient(double[] cost) {
        throw new NotImplementedException();
    }
}