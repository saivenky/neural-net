package saivenky.neural;

import saivenky.neural.activation.ActivationFunction;
import saivenky.neural.activation.Sigmoid;
import saivenky.neural.cost.CostFunction;
import saivenky.neural.cost.CrossEntropy;
import saivenky.neural.cost.Square;

/**
 * Created by saivenky on 1/26/17.
 */
public class NeuralNetwork {
    Layer[] layers;

    int trainedExamples;

    double[] predicted;
    CostFunction costFunction;
    Layer outputLayer;

    public NeuralNetwork(int[] layerSizes, ActivationFunction activationFunction, CostFunction costFunction) {
        layers = new Layer[layerSizes.length - 1];
        for(int i = 1; i < layerSizes.length; i++) {
            layers[i - 1] = new Layer(layerSizes[i], layerSizes[i-1], activationFunction);
        }

        outputLayer = layers[layers.length - 1];
        trainedExamples = 0;
        this.costFunction = costFunction;
    }

    public void run(double[] input) {
        for(int i = 0; i < layers.length; i++) {
            layers[i].run(input);
            input = layers[i].activation;
        }

        predicted = input;
    }

    private void backpropagate(double[] input, double[] output) {
        double[] cost = costFunction.f1(predicted, output);
        double[] error = new double[cost.length];
        Vector.multiply(cost, outputLayer.activation1, error);
        outputLayer.error = error;

        double[] previousLayerActivation;
        double[] previousLayerActivation1;

        for(int i = layers.length - 1; i >= 0; i--) {
            if (i == 0) {
                previousLayerActivation = input;
                previousLayerActivation1 = null;
            }
            else {
                previousLayerActivation = layers[i-1].activation;
                previousLayerActivation1 = layers[i-1].activation1;
            }

            layers[i].backpropagate(previousLayerActivation, previousLayerActivation1);

            if (i != 0) {
                layers[i-1].error = layers[i].previousLayerError;
            }
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

    public double loss(double[] input, double[] output, CostFunction lossFunction) {
        run(input);
        return lossFunction.f(predicted, output);
    }

    public static void main(String[] args) {
        int[] layers = {2, 8, 2, 1};
        Vector.initialize(System.currentTimeMillis());
        NeuralNetwork nn = new NeuralNetwork(layers, Sigmoid.getInstance(), CrossEntropy.getInstance());

        Data.Function function = new Data.Function() {
            @Override
            double f(double x) {
                return 9. / (3. + x);
            }
        };

        Data.Example[] trainData = Data.generateFunction(function, 250);
        int batchSize = 1;
        double learningRate = 0.3;

        for(int i = 0; i < 200; i++) {
            Data.shuffle(trainData);
            int batchEnd = batchSize - 1;
            for(int j = 0; j < trainData.length; j++) {
                Data.Example e = trainData[j];
                nn.train(e.input, e.output);
                if (j == batchEnd) {
                    nn.update(learningRate);
                    batchEnd += batchSize;
                    if (batchEnd >= trainData.length) batchEnd = trainData.length - 1;
                }
            }
            double trainLoss = totalLoss(nn, trainData);
            if (trainLoss < 0.001) {
                System.out.printf("iter: %d\ntrainLoss: %s\n", i+1, trainLoss);
                break;
            }
        }

        Data.Example[] testData = Data.generateFunction(function, 250);

        //should be greater than 0.8
        System.out.println("\ntest data correct: " + check1d(nn, testData));
        System.out.println("testLoss: " + totalLoss(nn, testData));
        System.out.println("trainLoss: " + totalLoss(nn, trainData));
    }

    private static double check1d(NeuralNetwork nn, Data.Example[] data) {
        double correct = 0;
        for(Data.Example e : data) {
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
            totalLoss += nn.loss(e.input, e.output, Square.getInstance());
        }

        return totalLoss / data.length;
    }

    private static boolean same(double a, double b) {
        return Math.abs(a - b) < 1e-5;
    }
}
