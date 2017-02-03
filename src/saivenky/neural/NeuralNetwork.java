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
    Layer[] layers;

    private int trainedExamples;

    public double[] predicted;
    private CostFunction costFunction;
    private Layer outputLayer;
    private InputLayer inputLayer;

    public NeuralNetwork(
            int[] layerSizes, ActivationFunction activationFunction, CostFunction costFunction, NeuronInitializer neuronInitializer) {
        this(new InputLayer(layerSizes[0]), costFunction, new Layer[layerSizes.length - 1]);
        NeuronSet inputNeuronSet = new NeuronSet(inputLayer.neurons);
        NeuronSet previousLayerNeurons = inputNeuronSet;
        for (int i = 1; i < layerSizes.length; i++) {
            layers[i - 1] = new StandardLayer(
                    layerSizes[i], previousLayerNeurons, activationFunction, neuronInitializer);
            previousLayerNeurons = layers[i-1].neurons;
        }

        outputLayer = layers[layers.length - 1];
    }

    public NeuralNetwork(InputLayer inputLayer, CostFunction costFunction, Layer ... layers) {
        this.inputLayer = inputLayer;
        this.costFunction = costFunction;
        this.layers = layers;
        outputLayer = layers[layers.length - 1];
        trainedExamples = 0;
    }

    public void setDropouts(double[] dropouts) {
        for(int i = 0; i < layers.length - 1; i++) {
            layers[i].setDropoutRate(dropouts[i]);
        }
    }

    public void run(double[] input) {
        inputLayer.setInput(input);
        double inputDropoutRate = 0;
        for (int i = 0; i < layers.length; i++) {
            layers[i].runScaled(inputDropoutRate);
            inputDropoutRate = layers[i].dropoutRate;
        }

        predicted = new double[outputLayer.neurons.size()];
        outputLayer.neurons.forAll(new NeuronSet.NeuronAction() {
            @Override
            void f(Neuron neuron, int i) {
                predicted[i] = neuron.activation;
            }
        });
    }

    private void feedforward(double[] input) {
        inputLayer.setInput(input);
        for (int i = 0; i < layers.length; i++) {
            layers[i].run();
        }

        predicted = new double[outputLayer.neurons.size()];
        outputLayer.neurons.forAll(new NeuronSet.NeuronAction() {
            @Override
            void f(Neuron neuron, int i) {
                predicted[i] = neuron.activation;
            }
        });
    }

    private void backpropagate(double[] output) {
        double[] cost = costFunction.f1(predicted, output);
        outputLayer.neurons.forAll(new NeuronSet.NeuronAction() {
            @Override
            void f(Neuron neuron, int i) {
                neuron.signalCostGradient += cost[i] * neuron.activation1;
            }
        });

        for (int i = layers.length - 1; i >= 1; i--) {
            layers[i].backpropagate();
        }

        for (int i = layers.length - 1; i >= 0; i--) {
            layers[i].updateGradient();
        }
    }

    void update(double learningRate) {
        for (int i = 0; i < layers.length; i++) {
            layers[i].gradientDescent(learningRate / trainedExamples);
        }

        trainedExamples = 0;
    }

    void train(double[] input, double[] output) {
        feedforward(input);
        backpropagate(output);
        trainedExamples += 1;
    }

    void reselectDropouts() {
        for (Layer l : layers) {
            l.reselectDropout();
        }
    }

    private double cost(double[] input, double[] output, CostFunction costFunction) {
        run(input);
        return costFunction.f(predicted, output);
    }

    public static void main(String[] args) {
        int[] layers = {2, 8, 2, 1};
        Vector.initialize(System.currentTimeMillis());
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

        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(nn, trainData);

        int batchSize = 1;
        double learningRate = 0.3;

        for (int i = 0; i < 1000; i++) {
            trainer.train(learningRate, batchSize, NullEvaluator);
            double trainLoss = totalLoss(nn, trainData);
            if (trainLoss < 0.001) {
                System.out.printf("iter: %d\ntrainLoss: %s\n", i + 1, trainLoss);
                break;
            }
        }

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
