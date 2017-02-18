package saivenky.neural;

import saivenky.neural.activation.ActivationFunction;
import saivenky.neural.neuron.NeuronInitializer;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * Created by saivenky on 2/1/17.
 */
public class ArrayImplConvolutionLayer implements ILayer {

    private final double[] signal;
    private final double[] inputError;
    private final double[] inputActivation;
    private final double[] outputError;
    private int kernelWidth;
    private int kernelHeight;
    private final int kernelDepth;
    private int frames;
    private final NeuronSet input;
    final NeuronProperties[] properties;
    private final int outputWidth;
    private final int outputHeight;
    private final int outputDepth;
    private int inputWidth;
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
        properties = new NeuronProperties[frames];
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;
        this.kernelDepth = input.getDepth();
        kernelWidthHeight = kernelWidth * kernelHeight;
        kernelWidthHeightDepth = kernelWidth * kernelHeight * kernelDepth;

        outputWidth = FilterDimensionCalculator.calculateOutputSize(input.getWidth(), kernelWidth, 1);
        outputHeight = FilterDimensionCalculator.calculateOutputSize(input.getHeight(), kernelHeight, 1);
        outputDepth = FilterDimensionCalculator.calculateOutputSize(input.getDepth(), kernelDepth, 1);
        outputWidthHeight = outputWidth * outputHeight;
        outputWidthHeightDepth = outputWidth * outputHeight * outputDepth;

        inputWidth = input.getWidth();
        inputWidthHeight = input.getWidth() * input.getHeight();

        neuronArray = new BasicNeuron[outputWidth * outputHeight * outputDepth * frames];
        neurons = new NeuronSet(neuronArray);
        for(int f = 0; f < frames; f++) {
            properties[f] = new NeuronProperties(neuronInitializer, kernelWidth * kernelHeight * input.getDepth());
            int frameStart = f * outputWidthHeightDepth;
            for(int i = 0; i < outputWidthHeightDepth; i++) {
                neuronArray[frameStart + i] = new BasicNeuron(activationFunction);
            }
        }

        neurons.setShape(outputWidth, outputHeight, frames);


        signal = new double[outputWidth * outputHeight * outputDepth * frames];
        outputError = new double[outputWidth * outputHeight * outputDepth * frames];
        inputError = new double[input.size()];
        inputActivation = new double[input.size()];
    }

    private void backpropogateNeuronError() {
        for (int frame = 0; frame < frames; frame++) {
            int frameStart = frame * outputWidthHeightDepth;
            double[] weights = properties[frame].getWeights();

            for (int outZ = 0, inZInit = 0; outZ < outputWidthHeightDepth; outZ += outputWidthHeight, inZInit += inputWidthHeight) {
                for (int outY = 0, inYInit = 0; outY < outputWidthHeight; outY += outputWidth, inYInit += inputWidth) {
                    for (int x = 0; x < outputWidth; x++) {
                        double neuronError = neuronArray[frameStart + x + outY + outZ].getError();

                        for (int kz = 0, inZ = inZInit, kernZ = 0; kz < kernelDepth; kz++, inZ += inputWidthHeight, kernZ += kernelWidthHeight) {
                            for (int ky = 0, inYZ = inZ + inYInit, kernY = 0; ky < kernelHeight; ky++, inYZ += input.getWidth(), kernY += kernelWidth) {
                                for (int kx = 0, inXYZ = inYZ + x; kx < kernelWidth; kx++, inXYZ++) {
                                    double weight = weights[kernZ + kernY + kx];
                                    inputError[inXYZ] += weight * neuronError;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private void copyInputActivation() {
        for(int i = 0; i < input.size(); i++) {
            inputActivation[i] = input.get(i).getActivation();
        }
    }

    private void copyOutputError() {
        for(int i = 0; i < neuronArray.length; i++) {
            outputError[i] = neuronArray[i].getError();
        }
    }


    private void applyConvolution() {
        Thread[] threads = new Thread[frames];
        for(int frame = 0; frame < frames; frame++) {
            int finalFrame = frame;
            threads[frame] = new Thread(() -> applyConvolutionForFrame(finalFrame));
            threads[frame].start();
        }

        for(Thread t : threads) {
            try {
                t.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e);
            }
        }
    }

    private void applyConvolutionForFrame(int frame) {
        int frameStart = frame * outputWidthHeightDepth;
        double[] weights = properties[frame].getWeights();
        double bias = properties[frame].getBias();
        for (int outZ = 0, inZInit = 0; outZ < outputWidthHeightDepth; outZ += outputWidthHeight, inZInit += inputWidthHeight) {
            for (int outY = 0, inYInit = 0; outY < outputWidthHeight; outY += outputWidth, inYInit += inputWidth) {
                for (int x = 0; x < outputWidth; x++) {

                    double sum = 0;
                    int outIndex = frameStart + outZ + outY + x;
                    signal[outIndex] = bias;
                    for (int kz = 0, inZ = inZInit, kernZ = 0; kz < kernelDepth; kz++, inZ += inputWidthHeight, kernZ += kernelWidthHeight) {
                        for (int ky = 0, inYZ = inZ + inYInit, kernY = 0; ky < kernelHeight; ky++, inYZ += input.getWidth(), kernY += kernelWidth) {
                            for (int kx = 0, inXYZ = inYZ + x; kx < kernelWidth; kx++, inXYZ++) {
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
        applyConvolution();
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

        copyOutputError();
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
            backpropogateToPropertiesForSingleFrame(frame);
        }
    }

    private void backpropogateToPropertiesForSingleFrame(int frame) {
        int frameStart = frame * outputWidthHeightDepth;
        NeuronProperties frameProperties = properties[frame];
        for (int outZ = 0, inZInit = 0; outZ < outputWidthHeightDepth; outZ += outputWidthHeight, inZInit += inputWidthHeight) {
            for (int outY = 0, inYInit = 0; outY < outputWidthHeight; outY += outputWidth, inYInit += inputWidth) {
                for (int x = 0; x < outputWidth; x++) {
                    double error = outputError[frameStart + outZ + outY + x];
                    frameProperties.biasCostGradient += error;

                    for (int kz = 0, inZ = inZInit, kernZ = 0; kz < kernelDepth; kz++, inZ += inputWidthHeight, kernZ += kernelWidthHeight) {
                        for (int ky = 0, inYZ = inZ + inYInit, kernY = 0; ky < kernelHeight; ky++, inYZ += input.getWidth(), kernY += kernelWidth) {
                            for (int kx = 0, inXYZ = inYZ + x; kx < kernelWidth; kx++, inXYZ++) {
                                double activation = inputActivation[inXYZ];
                                frameProperties.weightCostGradient[kernZ + kernY + kx] += activation * error;
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
