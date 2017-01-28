package saivenky.neural;

import java.util.Random;

/**
 * Created by saivenky on 1/26/17.
 */
public class NeuralNetwork {
    Layer[] layers;

    int trainedExamples;

    double[] predicted;

    public NeuralNetwork(int[] layerSizes) {
        layers = new Layer[layerSizes.length - 1];
        for(int i = 1; i < layerSizes.length; i++) {
            layers[i - 1] = new Layer(layerSizes[i], layerSizes[i-1]);
        }

        trainedExamples = 0;
    }

    public void run(double[] input) {
        double[] signal = input;
        for(int i = 0; i < layers.length; i++) {
            layers[i].computeActivation(signal);
            signal = layers[i].activation;
        }
        predicted = layers[layers.length - 1].activation;
    }

    private void backpropagate(double[] input, double[] output) {
        Layer outputLayer = layers[layers.length - 1];
        double[] error = new double[output.length];
        Vector.subtract(predicted, output, error);
        Vector.multiply(error, outputLayer.activation1, error);
        outputLayer.error = error;
        double[] previousLayerActivation = (layers.length > 1) ? layers[layers.length - 2].activation : input;
        for(int i = 0; i < outputLayer.neurons.length; i++) {
            Vector.multiplyAndAdd(previousLayerActivation, error[i], outputLayer.neurons[i].weightError);
        }

        for(int i = layers.length - 1; i > 0; i--) {
            layers[i].backpropagate(layers[i-1].activation, layers[i-1].activation1);
            layers[i-1].error = layers[i].previousLayerError;
        }
    }

    public void update(double learningRate) {
        for(int i = 0; i < layers.length; i++) {
            layers[i].update(learningRate / trainedExamples);
        }

        trainedExamples = 0;
    }

    public void train(double[] input, double[] output) {
        run(input);
        backpropagate(input, output);
        trainedExamples += 1;
    }

    public static void main(String[] args) {
        int[] layers = {2, 4, 4, 1};
        NeuralNetwork nn = new NeuralNetwork(layers);

        Data.Example[] trainData = Data.generateXor(250);

        for(int i = 0; i < 40; i++) {
            for(int j = 0; j < trainData.length; j++) {
                Data.Example e = trainData[j];
                nn.train(e.input, e.output);
            }

            nn.update(0.3);
        }

        Data.Example[] testData = Data.generateXor(100);

        System.out.println("test data correct: " + check1d(nn, testData));
    }

    private static double check1d(NeuralNetwork nn, Data.Example[] data) {
        double correct = 0;
        for(Data.Example e : data) {
            nn.run(e.input);
            double predicted = nn.predicted[0] > 0 ? 1 : -1;
            double actual = e.output[0];
            if (same(predicted, actual)) correct += 1;
        }

        return correct / data.length;
    }

    private static boolean same(double a, double b) {
        return Math.abs(a - b) < 1e-5;
    }
}
