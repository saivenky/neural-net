package saivenky.neural.c;

/**
 * Created by saivenky on 2/19/17.
 */
public class ReluLayer extends Layer {
    public ReluLayer(Layer previousLayer) {
        shape = previousLayer.shape;
        int size = previousLayer.shape[0] * previousLayer.shape[1] * previousLayer.shape[2];

        nativeLayerPtr = create(size, previousLayer.nativeLayerPtr);
        adjustByteOrderOnBuffers();
    }

    private native long create(int size, long previousLayerNativePtr);
    private native long destroy(long nativeLayerPtr);

    @Override
    public void gradientDescent(float rate) {
    }
}
