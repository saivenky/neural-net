package saivenky.neural.mnist;

import saivenky.neural.*;
import saivenky.neural.activation.Linear;
import saivenky.neural.activation.Sigmoid;
import saivenky.neural.c.FullyConnectedLayer;
import saivenky.neural.c.ReluLayer;
import saivenky.neural.c.SoftmaxCrossEntropyLayer;
import saivenky.neural.cost.CrossEntropy;
import saivenky.neural.image.ImageWriter;
import saivenky.neural.neuron.GaussianInitializer;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;

import static saivenky.neural.mnist.MnistReader.loadMnist;

/**
 * Created by saivenky on 2/6/17.
 */
public class MnistTester {
    private static final int IMAGE_WIDTH = 28;
    private static final int IMAGE_HEIGHT = 28;

    public static void main(String[] args) throws IOException {
        String trainImagesFilePath = args[0];
        String trainLabelsFilePath = args[1];
        String testImagesFilePath = args[2];
        String testLabelsFilePath = args[3];
        String outputDirectoryPath = args[4];

        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        System.out.print("Enter network type (standard, cnn, c-cnn, or small-cnn): ");
        String networkType = br.readLine().toLowerCase().intern();

        File outputDir = MnistReader.getRealFile(outputDirectoryPath);

        Data.Example[] trainingData = loadMnist(trainImagesFilePath, trainLabelsFilePath);
        Data.Example[] testData = loadMnist(testImagesFilePath, testLabelsFilePath);

        int[] testLabels = new int[testData.length];
        for(int i = 0; i < testLabels.length; i++) {
            testLabels[i] = argmax(testData[i].output);
        }

        System.out.print("Initializing network (");
        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(trainingData);
        trainer.setBatchSize(60);

        INeuralNetwork nn;
        switch(networkType) {
            case "standard":
                System.out.println("standard)");
                nn = getStandardNeuralNetwork(trainer);
                break;
            case "c-cnn":
                nn = getCConvolutionNeuralNetwork(trainer);
                System.out.println("cnn)");
                break;
            case "cnn":
                nn = getConvolutionNeuralNetwork(trainer);
                System.out.println("cnn)");
                break;
            case "small-cnn":
                nn = getSmallConvolutionNeuralNetwork(trainer);
                System.out.println("small-cnn)");
                break;
            default:
                nn = getStandardNeuralNetwork(trainer);
                System.out.println("default-standard)");
        }

        final boolean[] shouldEvaluateBatch = {true};

        NeuralNetworkTrainer.Evaluator epochEvaluator = new NeuralNetworkTrainer.Evaluator() {
            @Override
            public void f(int iteration, long timeTaken) {
                System.out.printf("Epoch %d complete (%.3fs). Accuracy: %s\n",
                        iteration, (double)timeTaken / 1000, checkLabels(nn, testData, testLabels, trainer.batchSize, testData.length));
            }
        };
        NeuralNetworkTrainer.Evaluator batchEvaluator = new NeuralNetworkTrainer.Evaluator() {
            @Override
            public void f(int iteration, long timeTaken) {
                if (timeTaken < 30) return;
                if (shouldEvaluateBatch[0]) {
                    double accuracy = checkLabels(nn, testData, testLabels, trainer.batchSize, 120);
                    System.out.printf("Batch %d complete (%.3fs). Accuracy: %s\n", iteration, (double) timeTaken / 1000, accuracy);
                    if (accuracy > 0.9) shouldEvaluateBatch[0] = false;
                } else {
                    System.out.printf("Batch %d complete (%.3fs).\n", iteration, (double) timeTaken / 1000);
                }
            }
        };
        trainer.train(epochEvaluator, batchEvaluator);
        trainer.setEpochs(1);
        trainer.setLearningRate(0.0001);
        trainer.train(epochEvaluator, batchEvaluator);

        System.out.println("test data correct: " + checkLabelsAndWriteIncorrect(nn, testData, testLabels, trainer.batchSize, outputDir));
    }

    private static INeuralNetwork getStandardNeuralNetwork(NeuralNetworkTrainer trainer) {
        int[] layers = {IMAGE_WIDTH * IMAGE_HEIGHT, 120, 10};
        double[] dropouts = {0, 0.25, 0};

        NeuralNetwork nn = new NeuralNetwork(layers, Sigmoid.getInstance(), CrossEntropy.getInstance(), GaussianInitializer.getInstance(), dropouts);
        System.out.println("...");

        trainer.setNeuralNetwork(nn);
        trainer.setLearningRate(3.0);
        trainer.setEpochs(15);

        return nn;
    }

    private static INeuralNetwork getCConvolutionNeuralNetwork(NeuralNetworkTrainer trainer) {
        saivenky.neural.c.InputLayer inputLayer = new saivenky.neural.c.InputLayer(
                IMAGE_WIDTH * IMAGE_HEIGHT, trainer.batchSize);
        inputLayer.setShape(IMAGE_WIDTH, IMAGE_HEIGHT, 1);
        int[] kernelShape = {5, 5, 1};
        int[] poolShape = {2, 2, 1};

        saivenky.neural.c.ConvolutionLayer convolutionLayer = new saivenky.neural.c.ConvolutionLayer(
                inputLayer, kernelShape, 20, 0);
        saivenky.neural.c.MaxPoolingLayer poolingLayer = new saivenky.neural.c.MaxPoolingLayer(
                convolutionLayer, poolShape, 2);
        saivenky.neural.c.ReluLayer reluLayer1 = new saivenky.neural.c.ReluLayer(poolingLayer);

        FullyConnectedLayer fcLayer1 = new FullyConnectedLayer(reluLayer1, 100);
        ReluLayer reluLayer2 = new ReluLayer(fcLayer1);
        FullyConnectedLayer fcLayer2 = new FullyConnectedLayer(reluLayer2, 10);
        SoftmaxCrossEntropyLayer outputLayer = new SoftmaxCrossEntropyLayer(fcLayer2, trainer.batchSize);
        saivenky.neural.c.NeuralNetwork nn = new saivenky.neural.c.NeuralNetwork(
                trainer.batchSize,
                inputLayer,
                convolutionLayer,
                poolingLayer,
                reluLayer1,
                fcLayer1,
                reluLayer2,
                fcLayer2,
                outputLayer);
        System.out.println("...");

        trainer.setNeuralNetwork(nn);
        trainer.setLearningRate(0.18);
        trainer.setEpochs(1);

        return nn;
    }

    private static INeuralNetwork getConvolutionNeuralNetwork(NeuralNetworkTrainer trainer) {
        NeuronSet imageNeurons = new NeuronSet(new INeuron[IMAGE_WIDTH * IMAGE_HEIGHT]);
        imageNeurons.setShape(IMAGE_WIDTH, IMAGE_HEIGHT, 1);
        InputLayer inputLayer = new InputLayer(imageNeurons);

        ConvolutionLayer convolutionLayer = new ConvolutionLayer(
                20, 7, 7, inputLayer, Sigmoid.getInstance(), GaussianInitializer.getInstance());
        MaxPoolingLayer poolingLayer = new MaxPoolingLayer(2, 2, convolutionLayer);

        StandardLayer standardLayer = new StandardLayer(
                100, poolingLayer, Sigmoid.getInstance(), GaussianInitializer.getInstance(), 0);
        StandardLayer outputLayer = new StandardLayer(
                10, standardLayer, Sigmoid.getInstance(), GaussianInitializer.getInstance(), 0);

        NeuralNetwork nn = new NeuralNetwork(inputLayer, CrossEntropy.getInstance(), 1,
                convolutionLayer, poolingLayer, standardLayer, outputLayer);
        System.out.println("...");

        trainer.setNeuralNetwork(nn);
        trainer.setLearningRate(0.1);
        trainer.setEpochs(200);

        return nn;
    }

    private static INeuralNetwork getSmallConvolutionNeuralNetwork(NeuralNetworkTrainer trainer) {
        NeuronSet imageNeurons = new NeuronSet(new INeuron[IMAGE_WIDTH * IMAGE_HEIGHT]);
        imageNeurons.setShape(IMAGE_WIDTH, IMAGE_HEIGHT, 1);
        InputLayer inputLayer = new InputLayer(imageNeurons);

        saivenky.neural.ConvolutionLayer convolutionLayer = new saivenky.neural.ConvolutionLayer(
                10, 14, 14, inputLayer, Linear.getInstance(), GaussianInitializer.getInstance());
        System.out.print(".");
        System.out.print(".");

        StandardLayer standardLayer = new StandardLayer(
                30, convolutionLayer, Sigmoid.getInstance(), GaussianInitializer.getInstance(), 0);
        StandardLayer outputLayer = new StandardLayer(
                10, standardLayer, Sigmoid.getInstance(), GaussianInitializer.getInstance(), 0);

        NeuralNetwork nn = new NeuralNetwork(inputLayer, CrossEntropy.getInstance(), 1, convolutionLayer, standardLayer, outputLayer);
        System.out.println(".");

        trainer.setNeuralNetwork(nn);
        trainer.setLearningRate(0.3);
        trainer.setEpochs(15);

        return nn;
    }

    private static double checkLabelsAndWriteIncorrect(INeuralNetwork nn, Data.Example[] data, int[] labels, int batchSize, File outputDir) {
        double correct = 0;
        if(data.length != labels.length) throw  new RuntimeException("Data and label length mismatch");

        if(!outputDir.exists()) outputDir.mkdir();

        File incorrectDirectory = new File(outputDir, "incorrect");
        File correctDirectory = new File(outputDir, "correct");
        if(!incorrectDirectory.exists()) incorrectDirectory.mkdir();
        if(!correctDirectory.exists()) correctDirectory.mkdir();

        double[][] input = new double[batchSize][];
        int[] labelIndex = new int[batchSize];
        int batchEnd = batchSize - 1;
        for(int i = 0; i < data.length; i++) {
            Data.Example e = data[i];
            input[i % batchSize] = e.input;
            labelIndex[i % batchSize] = i;
            if (i == batchEnd) {
                nn.run(input);
                for (int t = 0; t < batchSize; t++) {
                    double[] predicted = nn.getPredicted()[t];
                    int labelI = labelIndex[t];
                    int predictedLabel = argmax(predicted);
                    int expectedLabel = labels[labelI];
                    String filename = String.format("%7d-a%de%d.png", i, predictedLabel, expectedLabel);
                    File image;
                    if (predictedLabel == expectedLabel) {
                        correct += 1;
                        image = new File(correctDirectory, filename);
                    }
                    else {
                        image = new File(incorrectDirectory, filename);
                    }

                    ImageWriter.write(image, IMAGE_WIDTH, IMAGE_HEIGHT, e.input);
                }
                batchEnd += batchSize;
            }
        }

        return correct / data.length;
    }

    private static double checkLabels(
            INeuralNetwork nn, Data.Example[] data, int[] labels, int batchSize, int lengthToCheck) {
        double correct = 0;
        if(data.length != labels.length) throw  new RuntimeException("Data and label length mismatch");
        double[][] input = new double[batchSize][];
        int[] labelIndex = new int[batchSize];
        int batchEnd = batchSize - 1;
        for(int i = 0; i < lengthToCheck; i++) {
            Data.Example e = data[i];
            input[i % batchSize] = e.input;
            labelIndex[i % batchSize] = i;
            if (i == batchEnd) {
                nn.run(input);
                for (int t = 0; t < batchSize; t++) {
                    double[] predicted = nn.getPredicted()[t];
                    int labelI = labelIndex[t];
                    int predictedLabel = argmax(predicted);
                    if (predictedLabel == labels[labelI]) correct += 1;
                }
                batchEnd += batchSize;
            }
        }

        return correct / lengthToCheck;
    }

    private static int argmax(double[] array) {
        double max = -Double.MAX_VALUE;
        int indexOfMax = -1;
        for(int i = 0; i < array.length; i++) {
            if(array[i] > max) {
                max = array[i];
                indexOfMax = i;
            }
        }

        return indexOfMax;
    }
}
