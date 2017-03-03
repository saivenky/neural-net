package saivenky.neural;

import saivenky.neural.activation.ActivationFunction;
import saivenky.neural.activation.Sigmoid;
import saivenky.neural.cost.CostFunction;
import saivenky.neural.cost.CrossEntropy;
import saivenky.neural.cost.Square;
import saivenky.neural.neuron.NeuronInitializer;

import static saivenky.neural.NeuralNetworkTrainer.NullEvaluator;

/**
 * Created by saivenky on 1/26/17.
 */
public class NeuralNetwork {
    ILayer[] layers;

    private int trainedExamples;

    private IInputLayer inputLayer;
    private IOutputLayer outputLayer;
    private CostFunction costFunction;
    public double[][] predicted;

    public NeuralNetwork(
            int[] layerSizes, ActivationFunction activationFunction, CostFunction costFunction, NeuronInitializer neuronInitializer) {
        this(layerSizes, activationFunction, costFunction, neuronInitializer, new double[layerSizes.length]);
    }

    public NeuralNetwork(
            int[] layerSizes, ActivationFunction activationFunction, CostFunction costFunction, NeuronInitializer neuronInitializer, double[] dropoutRate) {
        this(new InputLayer(layerSizes[0]), costFunction, 1, new Layer[layerSizes.length - 1]);
        ILayer previousLayer = inputLayer;
        for (int i = 1; i < layerSizes.length; i++) {
            layers[i - 1] = new StandardLayer(
                    layerSizes[i], previousLayer, activationFunction, neuronInitializer, dropoutRate[i]);
            previousLayer = layers[i-1];
        }

        if (!(layers[layers.length - 1] instanceof IOutputLayer)) {
            throw new RuntimeException("no output layer");
        }
        outputLayer = (IOutputLayer) layers[layers.length - 1];
        predicted = new double[1][outputLayer.getNeurons().size()];
    }

    public NeuralNetwork(IInputLayer inputLayer, CostFunction costFunction, int miniBatchSize, ILayer ... layers) {
        this.inputLayer = inputLayer;
        this.costFunction = costFunction;
        this.layers = layers;
        if (!(layers[layers.length - 1] instanceof IOutputLayer)) {
            throw new RuntimeException("no output layer");
        }
        outputLayer = (IOutputLayer)layers[layers.length - 1];
        trainedExamples = 0;
        if (outputLayer != null) predicted = new double[miniBatchSize][outputLayer.size()];
    }

    public void run(double[][] input) {
        inputLayer.setInput(input);
        for (int i = 0; i < layers.length; i++) {
            layers[i].run();
        }

        outputLayer.getPredicted(predicted);
    }

    private void feedforward(double[][] input) {
        inputLayer.setInput(input);
        for (int i = 0; i < layers.length; i++) {
            layers[i].feedforward();
        }

        outputLayer.getPredicted(predicted);
    }

    private void backpropagate(double[][] output) {
        outputLayer.setExpected(output);
        double[][] cost = new double[output.length][];
        for(int i = 0; i < output.length; i++) {
            cost[i] = costFunction.f1(predicted[i], output[i]);
        }

        outputLayer.setSignalCostGradient(cost);

        for (int i = layers.length - 1; i >= 0; i--) {
            layers[i].backpropagate(i != 0); //input layer can't learn
        }
    }

    void update(double learningRate) {
        for (ILayer layer : layers) {
            layer.gradientDescent(learningRate / trainedExamples);
        }

        trainedExamples = 0;
    }

    void train(double[][] input, double[][] output) {
        feedforward(input);
        backpropagate(output);
        trainedExamples += 1;
    }

    void reselectDropouts() {
        for (ILayer l : layers) {
            if (l instanceof IDropoutLayer) {
                ((IDropoutLayer)l).reselectDropout();
            }
        }
    }
}
