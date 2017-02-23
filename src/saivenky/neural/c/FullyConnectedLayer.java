package saivenky.neural.c;

import saivenky.neural.NeuronSet;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.nio.ByteBuffer;

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
        inputActivation = previousLayer.outputSignal;
        inputError = previousLayer.outputError;

        nativeLayerPtr = create(inputSize, outputSize,
                inputActivation, inputError);
        System.out.printf("fc native ptr: 0x%x\n", nativeLayerPtr);
        adjustByteOrderOnBuffers();
    }

    private native long create(
            long inputSize, long outputSize,
            ByteBuffer inputActivation, ByteBuffer inputError);
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
