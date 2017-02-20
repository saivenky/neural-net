package saivenky.neural.c;

import saivenky.neural.BasicNeuron;
import saivenky.neural.NeuronSet;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Created by saivenky on 2/19/17.
 */
public class SigmoidLayer extends Layer {
    private final BasicNeuron[] neuronArray;
    private final NeuronSet neurons;

    public SigmoidLayer(Layer previousLayer, int size) {
        inputActivation = previousLayer.outputSignal;
        inputError = previousLayer.outputError;

        neuronArray = new BasicNeuron[size];
        neurons = new NeuronSet(neuronArray);
        for(int i = 0; i < neuronArray.length; i++) {
            neuronArray[i] = new BasicNeuron();
        }

        neurons.setShape(
                previousLayer.getNeurons().getWidth(),
                previousLayer.getNeurons().getHeight(),
                previousLayer.getNeurons().getDepth());

        nativeLayerPtr = create(size, inputActivation, inputError);
        ByteOrder nativeOrder = ByteOrder.nativeOrder();
        outputSignal.order(nativeOrder);
        outputError.order(nativeOrder);
    }

    private native long create(int size, ByteBuffer inputActivation, ByteBuffer inputError);
    private native long destroy(long nativeLayerPtr);
    private native void feedforward(long nativeLayerPtr);
    private native void backpropogate(long nativeLayerPtr);
    private native void update(long nativeLayerPtr, double rate);

    private void copyOutputError() {
        for(int i = 0, bbIndex = 0; i < neuronArray.length; i++, bbIndex += SIZEOF_DOUBLE) {
            outputError.putDouble(bbIndex, neuronArray[i].getError());
        }
    }

    private void copyOutputSignal() {
        for(int i = 0, bbIndex = 0; i < neuronArray.length; i++, bbIndex += SIZEOF_DOUBLE) {
            neuronArray[i].setSignal(outputSignal.getDouble(bbIndex));
        }
    }

    private void zeroError() {
        for(BasicNeuron basicNeuron : neuronArray) {
            basicNeuron.setSignalCostGradient(0);
        }
    }

    @Override
    public NeuronSet getNeurons() {
        return neurons;
    }

    @Override
    public void run() {
        feedforward();
    }

    @Override
    public void feedforward() {
        feedforward(nativeLayerPtr);
        copyOutputSignal();
    }

    @Override
    public void backpropagate(boolean backpropagateToPreviousLayer) {
        copyOutputError();
        backpropogate(nativeLayerPtr);
        zeroError();
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
