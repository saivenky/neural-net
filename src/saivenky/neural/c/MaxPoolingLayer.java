package saivenky.neural.c;

import saivenky.neural.FilterDimensionCalculator;

/**
 * Created by saivenky on 2/22/17.
 */
public class MaxPoolingLayer extends Layer {
    public MaxPoolingLayer(Layer previousLayer, int[] poolShape, int stride) {
        shape = new int[] {
                FilterDimensionCalculator.calculateOutputSize(previousLayer.shape[0], poolShape[0], stride, 0),
                FilterDimensionCalculator.calculateOutputSize(previousLayer.shape[1], poolShape[1], stride, 0),
                previousLayer.shape[2]
        };

        nativeLayerPtr = create(previousLayer.shape, poolShape, stride, previousLayer.nativeLayerPtr);
        adjustByteOrderOnBuffers();
    }

    private native long create(
            int[] inputShape, int[] poolShape, int stride, long previousLayerNativePtr);
    private native long destroy(long nativeLayerPtr);

    @Override
    public void gradientDescent(float rate) {
    }
}
