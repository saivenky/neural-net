package saivenky.neural.c;

import saivenky.neural.IInputLayer;
import saivenky.neural.INeuralNetwork;
import saivenky.neural.IOutputLayer;

/**
 * Created by saivenky on 3/4/17.
 */
public class NeuralNetwork implements INeuralNetwork {
    private final IInputLayer inputLayer;
    private final IOutputLayer outputLayer;
    private final long nativePtr;
    private float[][] predicted;
    private boolean hasPredictedChanged;
    private int trainedExample = 0;

    public NeuralNetwork(int miniBatchSize, Layer ... layers) {
        if (!(layers[0] instanceof IInputLayer)) {
            throw new RuntimeException("No input layer");
        }
        inputLayer = (IInputLayer)layers[0];

        if (!(layers[layers.length - 1] instanceof IOutputLayer)) {
            throw new RuntimeException("No output layer");
        }
        outputLayer = (IOutputLayer)layers[layers.length - 1];

        long[] nativeLayerPtrs = new long[layers.length];
        for (int i = 0; i < layers.length; i++) {
            nativeLayerPtrs[i] = layers[i].nativeLayerPtr;
        }

        hasPredictedChanged = false;
        predicted = new float[miniBatchSize][outputLayer.size()];
        nativePtr = create(nativeLayerPtrs);
    }

    public static native long create(long[] nativeLayerPtrs);
    public static native void run(long nativePtr);
    public static native void update(long nativePtr, float rate);
    public static native void train(long nativePtr);

    @Override
    public float[][] getPredicted() {
        if (hasPredictedChanged) {
            outputLayer.getPredicted(predicted);
            hasPredictedChanged = false;
        }
        return predicted;
    }

    @Override
    public void run(float[][] input) {
        inputLayer.setInput(input);
        run(nativePtr);
        hasPredictedChanged = true;
    }

    @Override
    public void update(float rate) {
        update(nativePtr, rate / (float)trainedExample);
        trainedExample = 0;
    }

    @Override
    public void train(float[][] input, float[][] output) {
        inputLayer.setInput(input);
        outputLayer.setExpected(output);
        train(nativePtr);
        hasPredictedChanged = true;
        trainedExample += input.length;
    }
}
