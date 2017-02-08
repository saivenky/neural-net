package saivenky.neural.mnist;

import saivenky.neural.*;
import saivenky.neural.activation.Linear;
import saivenky.neural.activation.Sigmoid;
import saivenky.neural.cost.CrossEntropy;
import saivenky.neural.image.ImageWriter;
import saivenky.neural.neuron.GaussianInitializer;

import java.io.File;
import java.io.IOException;

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
        File outputDir = MnistReader.getRealFile(outputDirectoryPath);

        Data.Example[] trainingData = loadMnist(trainImagesFilePath, trainLabelsFilePath);
        Data.Example[] testData = loadMnist(testImagesFilePath, testLabelsFilePath);

        int[] testLabels = new int[testData.length];
        for(int i = 0; i < testLabels.length; i++) {
            testLabels[i] = getLabel(testData[i].output);
        }

        System.out.print("Initializing network");
        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(trainingData);
        NeuralNetwork nn = getConvolutionNeuralNetwork(trainer);

        trainer.setBatchSize(60);
        final boolean[] shouldEvaluateBatch = {true};

        trainer.train(new NeuralNetworkTrainer.Evaluator() {
            @Override
            public void f(int iteration, long timeTaken) {
                System.out.printf("Epoch %d complete (%.3fs). Accuracy: %s\n", iteration, (double)timeTaken / 1000, checkLabels(nn, testData, testLabels, testData.length));
            }
        }, new NeuralNetworkTrainer.Evaluator() {
            @Override
            public void f(int iteration, long timeTaken) {
                if (shouldEvaluateBatch[0]) {
                    double accuracy = checkLabels(nn, testData, testLabels, 100);
                    System.out.printf("Batch %d complete (%.3fs). Accuracy: %s\n", iteration, (double)timeTaken / 1000, accuracy);
                    if (accuracy > 0.9) shouldEvaluateBatch[0] = false;
                }
                else {
                    System.out.printf("Batch %d complete (%.3fs).\n", iteration, (double)timeTaken / 1000);
                }
            }
        });

        System.out.println("test data correct: " + checkLabelsAndWriteIncorrect(nn, testData, testLabels, outputDir));
    }

    private static NeuralNetwork getStandardNeuralNetwork(NeuralNetworkTrainer trainer) {
        int[] layers = {IMAGE_WIDTH * IMAGE_HEIGHT, 120, 10};
        double[] dropouts = {0, 0.25, 0};

        NeuralNetwork nn = new NeuralNetwork(layers, Sigmoid.getInstance(), CrossEntropy.getInstance(), GaussianInitializer.getInstance(), dropouts);
        System.out.println("...");

        trainer.setNeuralNetwork(nn);
        trainer.setLearningRate(3.0);
        trainer.setEpochs(15);

        return nn;
    }

    private static NeuralNetwork getConvolutionNeuralNetwork(NeuralNetworkTrainer trainer) {
        NeuronSet imageNeurons = new NeuronSet(new INeuron[IMAGE_WIDTH * IMAGE_HEIGHT]);
        imageNeurons.setShape(IMAGE_WIDTH, IMAGE_HEIGHT, 1);
        InputLayer inputLayer = new InputLayer(imageNeurons);

        ConvolutionLayer convolutionLayer = new ConvolutionLayer(
                20, 7, 7, inputLayer, Sigmoid.getInstance(), GaussianInitializer.getInstance());
        System.out.print(".");

        MaxPoolingLayer poolingLayer = new MaxPoolingLayer(2, 2, convolutionLayer);
        System.out.print(".");

        StandardLayer standardLayer = new StandardLayer(
                100, poolingLayer, Sigmoid.getInstance(), GaussianInitializer.getInstance(), 0);
        StandardLayer outputLayer = new StandardLayer(
                10, standardLayer, Sigmoid.getInstance(), GaussianInitializer.getInstance(), 0);

        NeuralNetwork nn = new NeuralNetwork(inputLayer, CrossEntropy.getInstance(), convolutionLayer, poolingLayer, standardLayer, outputLayer);
        System.out.println(".");

        trainer.setNeuralNetwork(nn);
        trainer.setLearningRate(0.1);
        trainer.setEpochs(200);

        return nn;
    }

    private static NeuralNetwork getSmallConvolutionNeuralNetwork(NeuralNetworkTrainer trainer) {
        NeuronSet imageNeurons = new NeuronSet(new INeuron[IMAGE_WIDTH * IMAGE_HEIGHT]);
        imageNeurons.setShape(IMAGE_WIDTH, IMAGE_HEIGHT, 1);
        InputLayer inputLayer = new InputLayer(imageNeurons);

        ConvolutionLayer convolutionLayer = new ConvolutionLayer(
                10, 14, 14, inputLayer, Linear.getInstance(), GaussianInitializer.getInstance());
        System.out.print(".");
        System.out.print(".");

        StandardLayer standardLayer = new StandardLayer(
                30, convolutionLayer, Sigmoid.getInstance(), GaussianInitializer.getInstance(), 0);
        StandardLayer outputLayer = new StandardLayer(
                10, standardLayer, Sigmoid.getInstance(), GaussianInitializer.getInstance(), 0);

        NeuralNetwork nn = new NeuralNetwork(inputLayer, CrossEntropy.getInstance(), convolutionLayer, standardLayer, outputLayer);
        System.out.println(".");

        trainer.setNeuralNetwork(nn);
        trainer.setLearningRate(0.3);
        trainer.setEpochs(15);

        return nn;
    }

    private static double checkLabelsAndWriteIncorrect(NeuralNetwork nn, Data.Example[] data, int[] labels, File outputDir) {
        double correct = 0;
        if(data.length != labels.length) throw  new RuntimeException("Data and label length mismatch");

        if(!outputDir.exists()) outputDir.mkdir();

        File incorrectDirectory = new File(outputDir, "incorrect");
        File correctDirectory = new File(outputDir, "correct");
        if(!incorrectDirectory.exists()) incorrectDirectory.mkdir();
        if(!correctDirectory.exists()) correctDirectory.mkdir();

        for(int i = 0; i < data.length; i++) {
            Data.Example e = data[i];
            nn.run(e.input);
            int actual = getLabel(nn.predicted);
            int expected = labels[i];
            String filename = String.format("%7d-a%de%d.png", i, actual, expected);
            File image;
            if (actual == expected) {
                correct += 1;
                image = new File(correctDirectory, filename);
            }
            else {
                image = new File(incorrectDirectory, filename);
            }

            ImageWriter.write(image, IMAGE_WIDTH, IMAGE_HEIGHT, e.input);
        }

        return correct / data.length;
    }

    private static double checkLabels(NeuralNetwork nn, Data.Example[] data, int[] labels, int lengthToCheck) {
        double correct = 0;
        if(data.length != labels.length) throw  new RuntimeException("Data and label length mismatch");
        for(int i = 0; i < lengthToCheck; i++) {
            Data.Example e = data[i];
            nn.run(e.input);
            int predicted = getLabel(nn.predicted);
            if (predicted == labels[i]) correct += 1;
        }

        return correct / lengthToCheck;
    }

    private static int getLabel(double[] predicted) {
        double best = -1;
        int bestLabel = -1;
        for(int i = 0; i < predicted.length; i++) {
            if(predicted[i] > best) {
                best = predicted[i];
                bestLabel = i;
            }
        }

        return bestLabel;
    }
}
