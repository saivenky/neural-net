package saivenky.neural.c;

import saivenky.neural.FilterDimensionCalculator;
import saivenky.neural.NeuronSet;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.nio.ByteBuffer;

/**
 * Created by saivenky on 2/22/17.
 */
public class MaxPoolingLayer extends Layer {
    public MaxPoolingLayer(Layer previousLayer, int[] poolShape, int stride) {
        shape = new int[] {
                FilterDimensionCalculator.calculateOutputSize(previousLayer.shape[0], poolShape[0], stride),
                FilterDimensionCalculator.calculateOutputSize(previousLayer.shape[1], poolShape[1], stride),
                previousLayer.shape[2]
        };
        inputActivation = previousLayer.outputSignal;
        inputError = previousLayer.outputError;

        nativeLayerPtr = create(previousLayer.shape, poolShape, stride,
                inputActivation, inputError);
        adjustByteOrderOnBuffers();
    }

    private native long create(
            int[] inputShape, int[] poolShape, int stride,
            ByteBuffer inputActivation, ByteBuffer inputError);
    private native long destroy(long nativeLayerPtr);
    private native void feedforward(long nativeLayerPtr);
    private native void backpropogate(long nativeLayerPtr);

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

    @Override
    public void setSignalCostGradient(double[] cost) {
        throw new NotImplementedException();
    }
}
