package saivenky.neural;

import saivenky.neural.activation.ActivationFunction;
import saivenky.neural.neuron.NeuronInitializer;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * Created by saivenky on 2/1/17.
 */
public class ArrayImplConvolutionLayer implements ILayer {

    private final int outputFrameLength;
    private final double[] signal;
    private final double[] errors;
    private final double[] inputActivation;
    private int kernelWidth;
    private int kernelHeight;
    private final int kernelDepth;
    private int frames;
    private final NeuronSet input;
    final NeuronProperties[] properties;
    private final int outputWidth;
    private final int outputHeight;
    private final int outputDepth;
    private final BasicNeuron[] neuronArray;
    private final NeuronSet neurons;
    private int kernelWidthHeight;

    private int outputWidthHeight;
    private int inputWidthHeight;

    private int kernelWidthHeightDepth;
    private final int outputWidthHeightDepth;

    public ArrayImplConvolutionLayer(
            int frames, int kernelWidth, int kernelHeight, ILayer previousLayer, ActivationFunction activationFunction, NeuronInitializer neuronInitializer) {
        this.frames = frames;
        input = previousLayer.getNeurons();
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;
        this.kernelDepth = input.getDepth();
        properties = new NeuronProperties[frames];
        outputWidth = FilterDimensionCalculator.calculateOutputSize(input.getWidth(), kernelWidth, 1);
        outputHeight = FilterDimensionCalculator.calculateOutputSize(input.getHeight(), kernelHeight, 1);
        outputDepth = FilterDimensionCalculator.calculateOutputSize(input.getDepth(), kernelDepth, 1);

        neuronArray = new BasicNeuron[outputWidth * outputHeight * outputDepth * frames];
        neurons = new NeuronSet(neuronArray);
        outputFrameLength = outputWidth * outputHeight * outputDepth;
        for(int f = 0; f < frames; f++) {
            properties[f] = new NeuronProperties(neuronInitializer, kernelWidth * kernelHeight * input.getDepth());
            int frameStart = f * outputFrameLength;
            for(int i = 0; i < outputFrameLength; i++) {
                neuronArray[frameStart + i] = new BasicNeuron(activationFunction);
            }
        }

        neurons.setShape(outputWidth, outputHeight, frames);

        outputWidthHeight = outputWidth * outputHeight;
        inputWidthHeight = input.getWidth() * input.getHeight();
        kernelWidthHeight = kernelWidth * kernelHeight;
        kernelWidthHeightDepth = kernelWidth * kernelHeight * kernelDepth;
        outputWidthHeightDepth = outputWidth * outputHeight * outputDepth;
        signal = new double[outputWidth * outputHeight * outputDepth * frames];
        errors = new double[input.getWidth() * input.getHeight() * input.getDepth()];
        inputActivation = new double[input.size()];
    }

    private void backpropogateNeuronError() {
        for (int frame = 0; frame < frames; frame++) {
            int frameStart = frame * outputFrameLength;
            double[] weights = properties[frame].getWeights();

            for (int x = 0; x < outputWidth; x++) {
                for (int y = 0, outY = 0, inYInit = 0; y < outputHeight; y++, outY += outputWidth, inYInit += input.getWidth()) {
                    for (int z = 0, outZ = 0, inZInit = 0; z < outputDepth; z++, outZ += outputWidthHeight, inZInit += inputWidthHeight) {
                        double neuronError = neuronArray[frameStart + x + outY + outZ].getError();

                        for (int kx = 0, inX = x; kx < kernelWidth; kx++, inX++) {
                            for (int ky = 0, inXY = inX + inYInit, kernY = 0; ky < kernelHeight; ky++, inXY += input.getWidth(), kernY += kernelWidth) {
                                for (int kz = 0, inXYZ = inXY + inZInit, kernZ = 0; kz < kernelDepth; kz++, inXYZ += inputWidthHeight, kernZ += kernelWidthHeight) {
                                    double weight = weights[kernZ + kernY + kx];
                                    errors[inXYZ] += weight * neuronError;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private void applyConvolution(NeuronSet input) {
        for(int i = 0; i < input.size(); i++) {
            inputActivation[i] = input.get(i).getActivation();
        }

        for(int frame = 0; frame < frames; frame++) {
            int frameStart = frame * outputFrameLength;
            double[] weights = properties[frame].getWeights();
            double bias = properties[frame].getBias();

            for (int x = 0; x < outputWidth; x++) {
                for (int outY = 0, inYInit = 0; outY < outputWidthHeight; outY += outputWidth, inYInit += input.getWidth()) {
                    for (int outZ = 0, inZInit = 0; outZ < outputWidthHeightDepth; outZ += outputWidthHeight, inZInit += inputWidthHeight) {

                        double sum = 0;
                        int outIndex = frameStart + outZ + outY + x;
                        signal[outIndex] = bias;
                        for (int kx = 0, inX = x; kx < kernelWidth; kx++, inX++) {
                            for (int inXY = inX + inYInit, kernY = 0; kernY < kernelWidthHeight; inXY += input.getWidth(), kernY += kernelWidth) {
                                for (int inXYZ = inXY + inZInit, kernZ = 0; kernZ < kernelWidthHeightDepth; inXYZ += inputWidthHeight, kernZ += kernelWidthHeight) {
                                    double activation = inputActivation[inXYZ];
                                    double weight = weights[kernZ + kernY + kx];
                                    sum += activation * weight;
                                }
                            }
                        }

                        signal[outIndex] += sum;
                    }
                }
            }
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
        applyConvolution(input);
        for(int i = 0; i < neuronArray.length; i++) {
            neuronArray[i].setSignal(signal[i]);
        }

        for (BasicNeuron basicNeuron : neuronArray) {
            basicNeuron.activate();
        }
    }

    @Override
    public void backpropagate(boolean backpropagateToPreviousLayer) {
        for(BasicNeuron basicNeuron : neuronArray) {
            basicNeuron.backpropagate(false);
        }

        backpropogateToProperties();
        if(backpropagateToPreviousLayer) {
            backpropogateNeuronError();
            throw new RuntimeException("didn't do anything with error");
        }

        zeroError();
    }

    @Override
    public void gradientDescent(double rate) {
        for(int frame = 0; frame < frames; frame++) {
            properties[frame].update(rate);
        }
    }

    private void zeroError() {
        for(BasicNeuron basicNeuron : neuronArray) {
            basicNeuron.setSignalCostGradient(0);
        }
    }

    private void backpropogateToProperties() {
        for(int frame = 0; frame < frames; frame++) {
            int frameStart = frame * outputFrameLength;
            NeuronProperties frameProperties = properties[frame];
            for (int x = 0; x < outputWidth; x++) {
                for (int outY = 0, inYInit = 0; outY < outputWidthHeight; outY += outputWidth, inYInit += input.getWidth()) {
                    for (int outZ = 0, inZInit = 0; outZ < outputWidthHeightDepth; outZ += outputWidthHeight, inZInit += inputWidthHeight) {
                        double error = neuronArray[frameStart + outZ + outY + x].getError();
                        frameProperties.biasCostGradient += error;

                        for (int kx = 0, inX = x; kx < kernelWidth; kx++, inX++) {
                            for (int ky = 0, inXY = inX + inYInit, kernY = 0; ky < kernelHeight; ky++, inXY += input.getWidth(), kernY += kernelWidth) {
                                for (int kz = 0, inXYZ = inXY + inZInit, kernZ = 0; kz < kernelDepth; kz++, inXYZ += inputWidthHeight, kernZ += kernelWidthHeight) {
                                    double activation = inputActivation[inXYZ];
                                    frameProperties.weightCostGradient[kernZ + kernY + kx] += activation * error;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Override
    public void setSignalCostGradient(double[] cost) {
        throw new NotImplementedException();
    }
}
