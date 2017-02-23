package saivenky.neural.c;

import saivenky.neural.IOutputLayer;
import saivenky.neural.NeuronSet;

import java.nio.ByteBuffer;

/**
 * Created by saivenky on 2/20/17.
 */
public class OutputLayer extends Layer implements IOutputLayer {
    private final int size;

    public OutputLayer(Layer previousLayer) {
        shape = previousLayer.shape;
        size = previousLayer.shape[0] * previousLayer.shape[1] * previousLayer.shape[2];
        nativeLayerPtr = create(size, previousLayer.outputSignal, previousLayer.outputError);
        adjustByteOrderOnBuffers();
        outputSignal = inputActivation;
        outputError = inputError;
    }

    private native long create(int size, ByteBuffer inputActivation, ByteBuffer inputError);
    private native long destroy(long nativeLayerPtr);

    @Override
    public NeuronSet getNeurons() {
        return null;
    }

    @Override
    public void run() {
    }

    @Override
    public void feedforward() {
    }

    @Override
    public void backpropagate(boolean backpropagateToPreviousLayer) {
    }

    @Override
    public void gradientDescent(double rate) {
    }

    @Override
    public void setSignalCostGradient(double[] cost) {
        for (int i = 0, bbIndex = 0; i < cost.length; i++, bbIndex += SIZEOF_DOUBLE) {
            inputError.putDouble(bbIndex, cost[i]);
        }
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
