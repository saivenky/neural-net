package saivenky.neural.c;

import saivenky.neural.FilterDimensionCalculator;

/**
 * Created by saivenky on 2/16/17.
 */
public class ConvolutionLayer extends Layer {
    public ConvolutionLayer(Layer previousLayer, int[] kernelShapeWithoutDepth, int frames, int padding) {
        System.out.printf("Creating %s\n", this.getClass().toString());
        int[] kernelShape = { kernelShapeWithoutDepth[0], kernelShapeWithoutDepth[1], previousLayer.shape[2] };
        shape = new int[] {
            FilterDimensionCalculator.calculateOutputSize(previousLayer.shape[0], kernelShape[0], 1, padding),
            FilterDimensionCalculator.calculateOutputSize(previousLayer.shape[1], kernelShape[1], 1, padding),
            frames
        };

        nativeLayerPtr = create(
                previousLayer.shape, kernelShape, frames, 1, padding, previousLayer.nativeLayerPtr);
        adjustByteOrderOnBuffers();
    }

    private native long create(
            int[] inputShape, int[] kernelShape, int frames, int stride, int padding, long previousLayerNativePtr);
    private native long destroy(long nativeLayerPtr);
}
