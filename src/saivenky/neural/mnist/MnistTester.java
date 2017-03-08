package saivenky.neural.mnist;

import saivenky.neural.*;
import saivenky.neural.c.*;
import saivenky.neural.image.ImageWriter;

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
        System.out.print("Enter network type (standard, cnn): ");
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
            case "cnn":
                nn = getConvolutionNeuralNetwork(trainer);
                System.out.println("cnn)");
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
                if (iteration % 100 == 0) {
                    if (shouldEvaluateBatch[0]) {
                        double accuracy = checkLabels(nn, testData, testLabels, trainer.batchSize, 120);
                        System.out.printf("Batch %d complete (%.3fs). Accuracy: %s\n", iteration, (double) timeTaken / 1000, accuracy);
                        if (accuracy > 0.9) shouldEvaluateBatch[0] = false;
                    } else {
                        System.out.printf("Batch %d complete (%.3fs).\n", iteration, (double) timeTaken / 1000);
                    }
                }
            }
        };
        trainer.train(epochEvaluator, batchEvaluator);
        trainer.setEpochs(1);
        trainer.setLearningRate(0.0001f);
        trainer.train(epochEvaluator, batchEvaluator);

        System.out.println("test data correct: " + checkLabelsAndWriteIncorrect(nn, testData, testLabels, trainer.batchSize, outputDir));
    }

    private static INeuralNetwork getStandardNeuralNetwork(NeuralNetworkTrainer trainer) {
        InputLayer inputLayer = new InputLayer(IMAGE_WIDTH * IMAGE_HEIGHT, trainer.batchSize);
        inputLayer.setShape(IMAGE_WIDTH, IMAGE_HEIGHT, 1);

        FullyConnectedLayer fcLayer1 = new FullyConnectedLayer(inputLayer, 100);
        SigmoidLayer sigmoidLayer1 = new SigmoidLayer(fcLayer1);

        FullyConnectedLayer fcLayer2 = new FullyConnectedLayer(sigmoidLayer1, 10);
        SoftmaxCrossEntropyLayer softmaxCrossEntropyLayer = new SoftmaxCrossEntropyLayer(fcLayer2, trainer.batchSize);

        NeuralNetwork nn = new NeuralNetwork(
                trainer.batchSize,
                inputLayer,
                fcLayer1,
                sigmoidLayer1,
                fcLayer2,
                softmaxCrossEntropyLayer);
        System.out.println("...");

        trainer.setNeuralNetwork(nn);
        trainer.setLearningRate(3.0f);
        trainer.setEpochs(15);

        return nn;
    }

    private static INeuralNetwork getConvolutionNeuralNetwork(NeuralNetworkTrainer trainer) {
        InputLayer inputLayer = new InputLayer(IMAGE_WIDTH * IMAGE_HEIGHT, trainer.batchSize);
        inputLayer.setShape(IMAGE_WIDTH, IMAGE_HEIGHT, 1);

        int[] kernelShape = {5, 5, 1};
        int[] poolShape = {2, 2, 1};
        ConvolutionLayer convolutionLayer = new ConvolutionLayer(inputLayer, kernelShape, 20, 0);
        MaxPoolingLayer poolingLayer = new MaxPoolingLayer(convolutionLayer, poolShape, 2);
        ReluLayer reluLayer1 = new ReluLayer(poolingLayer);

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
        trainer.setLearningRate(0.18f);
        trainer.setEpochs(1);

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

        float[][] input = new float[batchSize][];
        int[] labelIndex = new int[batchSize];
        int batchEnd = batchSize - 1;
        for(int i = 0; i < data.length; i++) {
            Data.Example e = data[i];
            input[i % batchSize] = e.input;
            labelIndex[i % batchSize] = i;
            if (i == batchEnd) {
                nn.run(input);
                for (int t = 0; t < batchSize; t++) {
                    float[] predicted = nn.getPredicted()[t];
                    int labelI = labelIndex[t];
                    int predictedLabel = argmax(predicted);
                    int expectedLabel = labels[labelI];
                    String filename = String.format("%7d-a%de%d.png", labelI, predictedLabel, expectedLabel);
                    File image;
                    if (predictedLabel == expectedLabel) {
                        correct += 1;
                        image = new File(correctDirectory, filename);
                    }
                    else {
                        image = new File(incorrectDirectory, filename);
                    }

                    ImageWriter.write(image, IMAGE_WIDTH, IMAGE_HEIGHT, input[t]);
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
        float[][] input = new float[batchSize][];
        int[] labelIndex = new int[batchSize];
        int batchEnd = batchSize - 1;
        for(int i = 0; i < lengthToCheck; i++) {
            Data.Example e = data[i];
            input[i % batchSize] = e.input;
            labelIndex[i % batchSize] = i;
            if (i == batchEnd) {
                nn.run(input);
                for (int t = 0; t < batchSize; t++) {
                    float[] predicted = nn.getPredicted()[t];
                    int labelI = labelIndex[t];
                    int predictedLabel = argmax(predicted);
                    if (predictedLabel == labels[labelI]) correct += 1;
                }
                batchEnd += batchSize;
            }
        }

        return correct / lengthToCheck;
    }

    private static int argmax(float[] array) {
        float max = -Float.MAX_VALUE;
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
