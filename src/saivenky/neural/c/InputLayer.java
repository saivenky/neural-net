package saivenky.neural.c;

import saivenky.neural.IInputLayer;

import java.nio.ByteBuffer;

/**
 * Created by saivenky on 2/20/17.
 */
public class InputLayer extends Layer implements IInputLayer {
    public InputLayer(int size, int miniBatchSize) {
        shape = new int[] {size, 1, 1};
        outputSignals = new ByteBuffer[miniBatchSize];
        nativeLayerPtr = create(size, miniBatchSize);
        adjustByteOrderOnBuffers();
    }

    private native long create(int size, int miniBatchSize);
    private native long destroy(long nativeLayerPtr);

    public void setInput(double[][] input) {
        for(int b = 0; b < input.length; b++) {
            for (int i = 0, bbIndex = 0; i < input[b].length; i++, bbIndex += SIZEOF_DOUBLE) {
                outputSignals[b].putDouble(bbIndex, input[b][i]);
            }
        }
    }

    public void setShape(int width, int height, int depth) {
        shape = new int[] {width, height, depth};
    }

    @Override
    public void feedforward() {
    }

    @Override
    public void backpropagate() {
    }

    @Override
    public void gradientDescent(double rate) {
    }
}
