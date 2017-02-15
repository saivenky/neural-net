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

    public double[] predicted;
    private CostFunction costFunction;
    private ILayer outputLayer;
    private InputLayer inputLayer;

    public NeuralNetwork(
            int[] layerSizes, ActivationFunction activationFunction, CostFunction costFunction, NeuronInitializer neuronInitializer) {
        this(layerSizes, activationFunction, costFunction, neuronInitializer, new double[layerSizes.length]);
    }

    public NeuralNetwork(
            int[] layerSizes, ActivationFunction activationFunction, CostFunction costFunction, NeuronInitializer neuronInitializer, double[] dropoutRate) {
        this(new InputLayer(layerSizes[0]), costFunction, new Layer[layerSizes.length - 1]);
        ILayer previousLayer = inputLayer;
        for (int i = 1; i < layerSizes.length; i++) {
            layers[i - 1] = new StandardLayer(
                    layerSizes[i], previousLayer, activationFunction, neuronInitializer, dropoutRate[i]);
            previousLayer = layers[i-1];
        }

        outputLayer = layers[layers.length - 1];
        predicted = new double[outputLayer.getNeurons().size()];
    }

    private void updatePredicted() {
        for(int i = 0; i < outputLayer.getNeurons().size(); i++) {
            predicted[i] = outputLayer.getNeurons().get(i).getActivation();
        }
    }

    public NeuralNetwork(InputLayer inputLayer, CostFunction costFunction, ILayer ... layers) {
        this.inputLayer = inputLayer;
        this.costFunction = costFunction;
        this.layers = layers;
        outputLayer = layers[layers.length - 1];
        trainedExamples = 0;
        if (outputLayer != null) predicted = new double[outputLayer.getNeurons().size()];
    }

    public void run(double[] input) {
        inputLayer.setInput(input);
        for (int i = 0; i < layers.length; i++) {
            layers[i].run();
        }

        updatePredicted();
    }

    private void feedforward(double[] input) {
        inputLayer.setInput(input);
        for (int i = 0; i < layers.length; i++) {
            layers[i].feedforward();
        }

        updatePredicted();
    }

    private void backpropagate(double[] output) {
        double[] cost = costFunction.f1(predicted, output);
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

    void train(double[] input, double[] output) {
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

    private double cost(double[] input, double[] output, CostFunction costFunction) {
        run(input);
        return costFunction.f(predicted, output);
    }

    public static void main(String[] args) {
        int[] layers = {2, 8, 2, 1};
        NeuronInitializer neuronInitializer = new NeuronInitializer(
                new NeuronInitializer.Function() {
                    @Override
                    public double f() {
                        return 0.1;
                    }
                },
                new NeuronInitializer.Function() {
                    @Override
                    public double f() {
                        return Math.random() - 0.5;
                    }
        });

        NeuralNetwork nn = new NeuralNetwork(layers, Sigmoid.getInstance(), CrossEntropy.getInstance(), neuronInitializer);

        Data.Function function = new Data.Function() {
            @Override
            double f(double x) {
                return 9. / (3. + x);
            }
        };

        Data.Example[] trainData = Data.generateFunction(function, 250);
        Data.Example[] testData = Data.generateFunction(function, 250);

        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(trainData);
        trainer.setNeuralNetwork(nn);
        trainer.setBatchSize(1);
        trainer.setLearningRate(0.3);
        trainer.setEpochs(50);

        trainer.train(NullEvaluator, NullEvaluator);

        //should be greater than 0.8
        System.out.println("\ntest data correct: " + check1d(nn, testData));
        System.out.println("testLoss: " + totalLoss(nn, testData));
        System.out.println("trainLoss: " + totalLoss(nn, trainData));
    }

    private static double check1d(NeuralNetwork nn, Data.Example[] data) {
        double correct = 0;
        for (Data.Example e : data) {
            nn.run(e.input);
            if (nn.predicted.length != 1 && e.output.length != 1) {
                throw new RuntimeException("Not 1-d data");
            }
            double predicted = nn.predicted[0] > 0.5 ? 1 : 0;
            double actual = e.output[0];
            if (same(predicted, actual)) correct += 1;
        }

        return correct / data.length;
    }

    private static double totalLoss(NeuralNetwork nn, Data.Example[] data) {
        double totalLoss = 0;
        for (Data.Example e : data) {
            totalLoss += nn.cost(e.input, e.output, Square.getInstance());
        }

        return totalLoss / data.length;
    }

    private static boolean same(double a, double b) {
        return Math.abs(a - b) < 1e-5;
    }
}
