package saivenky.neural.c;

import saivenky.neural.IInputLayer;
import saivenky.neural.NeuronSet;

/**
 * Created by saivenky on 2/20/17.
 */
public class InputLayer extends Layer implements IInputLayer {
    public InputLayer(int size) {
        shape = new int[] {size, 1, 1};
        nativeLayerPtr = create(size);
        adjustByteOrderOnBuffers();
    }

    private native long create(int size);
    private native long destroy(long nativeLayerPtr);

    @Override
    public NeuronSet getNeurons() {
        return null;
    }

    public void setInput(double[] input) {
        for(int i = 0, bbIndex = 0; i < input.length; i++, bbIndex += SIZEOF_DOUBLE) {
            outputSignal.putDouble(bbIndex, input[i]);
        }
    }

    public void setShape(int width, int height, int depth) {
        shape = new int[] {width, height, depth};
    }

    @Override
    public void run() {
        feedforward();
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
    }
}
