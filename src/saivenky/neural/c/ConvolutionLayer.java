package saivenky.neural.c;

import saivenky.neural.FilterDimensionCalculator;
import saivenky.neural.NeuronSet;

import java.nio.ByteBuffer;

/**
 * Created by saivenky on 2/16/17.
 */
public class ConvolutionLayer extends Layer {
    public ConvolutionLayer(Layer previousLayer, int[] kernelShapeWithoutDepth, int frames) {
        System.out.printf("Creating %s\n", this.getClass().toString());
        int[] kernelShape = { kernelShapeWithoutDepth[0], kernelShapeWithoutDepth[1], previousLayer.shape[2] };
        shape = new int[] {
            FilterDimensionCalculator.calculateOutputSize(previousLayer.shape[0], kernelShape[0], 1),
            FilterDimensionCalculator.calculateOutputSize(previousLayer.shape[1], kernelShape[1], 1),
            frames
        };

        inputActivation = previousLayer.outputSignal;
        inputError = previousLayer.outputError;

        nativeLayerPtr = create(previousLayer.shape, kernelShape, frames, 1, inputActivation, inputError);
        adjustByteOrderOnBuffers();
    }

    private native long create(
            int[] inputShape, int[] kernelShape, int frames, int stride,
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
}
