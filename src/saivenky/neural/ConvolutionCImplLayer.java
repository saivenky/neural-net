package saivenky.neural;

import saivenky.neural.activation.ActivationFunction;
import saivenky.neural.neuron.NeuronInitializer;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Created by saivenky on 2/16/17.
 */
public class ConvolutionCImplLayer implements ILayer {
    private static final int SIZEOF_DOUBLE = 8;

    static {
        System.loadLibrary("neuron");
        System.out.println("Loaded 'neuron' library");
    }

    private ByteBuffer inputActivation;
    private ByteBuffer inputError;
    private ByteBuffer outputSignal;
    private ByteBuffer outputError;

    private final NeuronSet input;
    private final BasicNeuron[] neuronArray;
    private final NeuronSet neurons;

    private final long nativeLayerPtr;

    public ConvolutionCImplLayer(ILayer previousLayer, int[] kernelShapeWithoutDepth, int frames,
                                 ActivationFunction activationFunction, NeuronInitializer neuronInitializer) {
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
            neuronArray[i] = new BasicNeuron(activationFunction);
        }

        neurons.setShape(outputWidth, outputHeight, frames);

        nativeLayerPtr = createNativeLayer(inputShape, kernelShape, frames, 1);
        ByteOrder nativeOrder = ByteOrder.nativeOrder();
        inputActivation.order(nativeOrder);
        inputError.order(nativeOrder);
        outputSignal.order(nativeOrder);
        outputError.order(nativeOrder);
    }

    private native long createNativeLayer(int[] inputShape, int[] kernelShape, int frames, int stride);
    private native void applyConvolution(long nativeLayerPtr);
    private native void backpropogateToProperties(long nativeLayerPtr);
    private native void updateProperties(long nativeLayerPtr, double rate);

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

    private void applyActivationFunction() {
        for (BasicNeuron basicNeuron : neuronArray) {
            basicNeuron.activate();
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
        applyConvolution(nativeLayerPtr);
        copyOutputSignal();
        applyActivationFunction();
    }

    @Override
    public void backpropagate(boolean backpropagateToPreviousLayer) {
        for(BasicNeuron basicNeuron : neuronArray) {
            basicNeuron.backpropagate(false);
        }

        copyOutputError();
        backpropogateToProperties(nativeLayerPtr);

        if(backpropagateToPreviousLayer) {
            throw new NotImplementedException();
        }

        zeroError();
    }

    @Override
    public void gradientDescent(double rate) {
        updateProperties(nativeLayerPtr, rate);
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
