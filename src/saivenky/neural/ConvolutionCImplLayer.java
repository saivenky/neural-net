package saivenky.neural;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Created by saivenky on 2/16/17.
 */
public class ConvolutionCImplLayer implements ILayer {
    private static final int SIZEOF_DOUBLE = 8;

    static {
        System.loadLibrary("neural");
        System.out.println("Loaded 'neural' library");
    }

    private ByteBuffer inputActivation;
    private ByteBuffer inputError;
    private ByteBuffer outputSignal;
    private ByteBuffer outputError;

    private final NeuronSet input;
    private final BasicNeuron[] neuronArray;
    private final NeuronSet neurons;

    private final long nativeLayerPtr;

    public ConvolutionCImplLayer(ILayer previousLayer, int[] kernelShapeWithoutDepth, int frames) {
        System.out.printf("Creating %s\n", this.getClass().toString());
        input = previousLayer.getNeurons();
        int[] kernelShape = { kernelShapeWithoutDepth[0], kernelShapeWithoutDepth[1], input.getDepth() };
        int outputWidth = FilterDimensionCalculator.calculateOutputSize(input.getWidth(), kernelShape[0], 1);
        int outputHeight = FilterDimensionCalculator.calculateOutputSize(input.getHeight(), kernelShape[1], 1);
        int outputDepth = FilterDimensionCalculator.calculateOutputSize(input.getDepth(), input.getDepth(), 1);

        int[] inputShape = { input.getWidth(), input.getHeight(), input.getDepth() };

        neuronArray = new BasicNeuron[outputWidth * outputHeight * outputDepth * frames];
        neurons = new NeuronSet(neuronArray);
        for(int i = 0; i < neuronArray.length; i++) {
            neuronArray[i] = new BasicNeuron();
        }

        neurons.setShape(outputWidth, outputHeight, frames);

        inputActivation = ByteBuffer.allocateDirect(SIZEOF_DOUBLE * input.size());
        inputError = null;
        nativeLayerPtr = create(inputShape, kernelShape, frames, 1, inputActivation, inputError);

        ByteOrder nativeOrder = ByteOrder.nativeOrder();
        inputActivation.order(nativeOrder);
        //inputError.order(nativeOrder);
        outputSignal.order(nativeOrder);
        outputError.order(nativeOrder);
    }

    private native long create(
            int[] inputShape, int[] kernelShape, int frames, int stride,
            ByteBuffer inputActivation, ByteBuffer inputError);
    private native long destroy(long nativeLayerPtr);
    private native void feedforward(long nativeLayerPtr);
    private native void backpropogate(long nativeLayerPtr);
    private native void update(long nativeLayerPtr, double rate);

    private void copyInputActivation() {
        for(int i = 0, bbIndex = 0; i < input.size(); i++, bbIndex += SIZEOF_DOUBLE) {
            inputActivation.putDouble(bbIndex, input.get(i).getActivation());
        }
    }

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
        copyInputActivation();
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

    private void zeroError() {
        for(BasicNeuron basicNeuron : neuronArray) {
            basicNeuron.setSignalCostGradient(0);
        }
    }

    @Override
    public void setSignalCostGradient(double[] cost) {
        throw new NotImplementedException();
    }
}
