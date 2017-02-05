package saivenky.neural.mnist;

import saivenky.neural.*;
import saivenky.neural.activation.Linear;
import saivenky.neural.activation.Sigmoid;
import saivenky.neural.cost.CrossEntropy;
import saivenky.neural.image.DataUtils;
import saivenky.neural.image.ImageWriter;
import saivenky.neural.neuron.GaussianInitializer;

import java.io.*;

/**
 * Created by saivenky on 1/28/17.
 */
public class MnistReader {
    public static double[] readImage(InputStream stream, int rows, int columns) {
        int totalBytes = rows * columns;
        double[] imagePixels = new double[totalBytes];
        byte[] rawBytes = new byte[totalBytes];
        int numBytesRead = 0;

        try {
            numBytesRead = stream.read(rawBytes);
        } catch (IOException e) {
            e.printStackTrace();
        }

        if (numBytesRead != totalBytes) {
            throw new RuntimeException("Incorrect pixels when reading image");
        }

        for(int i = 0; i < totalBytes; i++) {
            imagePixels[i] = DataUtils.toPixel(rawBytes[i]);
        }

        return imagePixels;
    }

    public static double[][] readImages(File imagesFile) throws IOException {
        DataInputStream dis = new DataInputStream(new FileInputStream(imagesFile));
        int magicNum = dis.readInt();
        int numImages = dis.readInt();
        int rows = dis.readInt();
        int columns = dis.readInt();

        double[][] images = new double[numImages][];
        for(int i = 0; i < numImages; i++) {
            images[i] = readImage(dis, rows, columns);
        }

        return images;
    }

    public static int[] readLabels(File labelsFile) throws IOException {
        DataInputStream dis = new DataInputStream(new FileInputStream(labelsFile));
        int magicNum = dis.readInt();
        int numLabels = dis.readInt();

        int[] labels = new int[numLabels];
        for(int i = 0; i < numLabels; i++) {
            labels[i] = dis.readUnsignedByte();
        }

        return labels;
    }

    private static double[] toOutput(int label) {
        double[] output = new double[10];
        output[label] = 1;
        return output;
    }

    public static Data.Example[] loadMnist(File imagesFile, File labelsFile) throws IOException {
        System.out.print("loading MNIST");
        double[][] images = readImages(imagesFile);
        System.out.print(".");
        int[] labels = readLabels(labelsFile);
        System.out.print(".");
        Data.Example[] examples = new Data.Example[images.length];
        for(int i = 0; i < examples.length; i++) {
            examples[i] = new Data.Example(images[i], toOutput(labels[i]));
        }

        System.out.println(".");

        return examples;
    }

    public static void main(String[] args) throws IOException {
        Data.Example[] trainingData = loadMnist(
                new File("/home/saivenky/downloads/train-images.idx3-ubyte"),
                new File("/home/saivenky/downloads/train-labels.idx1-ubyte")
        );

        Data.Example[] testData = loadMnist(
                new File("/home/saivenky/downloads/t10k-images.idx3-ubyte"),
                new File("/home/saivenky/downloads/t10k-labels.idx1-ubyte")
        );

        int[] testLabels = new int[testData.length];
        for(int i = 0; i < testLabels.length; i++) {
            testLabels[i] = getLabel(testData[i].output);
        }

        System.out.print("Initializing network");
        NeuralNetwork nn = getConvolutionNeuralNetwork();

        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(nn, trainingData);

        int batchSize = 60;
        double learningRate = 0.1;
        int epochs = 200;

        for(int i = 0; i < epochs; i++) {
            System.out.println("Epoch " + i);
            trainer.train(learningRate, batchSize, new NeuralNetworkTrainer.Evaluator() {
                @Override
                public void f(int batchNumber, long batchTimeMillis) {
                    System.out.printf("Batch %d (%.3fs) - test data correct: %s\n", batchNumber, (double)batchTimeMillis / 1000, checkLabels(nn, testData, testLabels, 100));
                }
            });
            System.out.println("test data correct: " + checkLabels(nn, testData, testLabels, testData.length));
        }

        File outputDir = new File("/home/saivenky/mnist-testing-result");
        System.out.println("test data correct: " + checkLabelsAndWriteIncorrect(nn, testData, testLabels, outputDir));
    }

    private static final int IMAGE_WIDTH = 28;
    private static final int IMAGE_HEIGHT = 28;

    private static NeuralNetwork getStandardNeuralNetwork() {
        int[] layers = {IMAGE_WIDTH * IMAGE_HEIGHT, 120, 10};
        double[] dropouts = {0, 0.25, 0};

        NeuralNetwork nn = new NeuralNetwork(layers, Sigmoid.getInstance(), CrossEntropy.getInstance(), new GaussianInitializer(), dropouts);
        System.out.println("...");

        return nn;
    }

    private static NeuralNetwork getConvolutionNeuralNetwork() {
        Spatial2DStructure spatial2DStructure = new Spatial2DStructure(new INeuron[IMAGE_WIDTH * IMAGE_HEIGHT], IMAGE_WIDTH, IMAGE_HEIGHT);
        InputLayer inputLayer = new InputLayer(spatial2DStructure);

        ConvolutionLayer convolutionLayer = new ConvolutionLayer(
                Sigmoid.getInstance(), 20, 8, 8, inputLayer, spatial2DStructure);
        System.out.print(".");

        MaxPoolingLayer poolingLayer = new MaxPoolingLayer(2, 2, convolutionLayer);
        System.out.print(".");

        StandardLayer standardLayer = new StandardLayer(
                100, poolingLayer, Sigmoid.getInstance(), new GaussianInitializer(), 0);
        StandardLayer outputLayer = new StandardLayer(
                10, standardLayer, Sigmoid.getInstance(), new GaussianInitializer(), 0);

        NeuralNetwork nn = new NeuralNetwork(inputLayer, CrossEntropy.getInstance(), convolutionLayer, poolingLayer, standardLayer, outputLayer);
        System.out.println(".");

        return nn;
    }

    private static NeuralNetwork getSmallConvolutionNeuralNetwork() {
        Spatial2DStructure spatial2DStructure = new Spatial2DStructure(new INeuron[IMAGE_HEIGHT * IMAGE_HEIGHT], IMAGE_WIDTH, IMAGE_HEIGHT);
        InputLayer inputLayer = new InputLayer(spatial2DStructure);

        ConvolutionLayer convolutionLayer = new ConvolutionLayer(
                Linear.getInstance(), 10, 14, 14, inputLayer, spatial2DStructure);
        System.out.print(".");
        System.out.print(".");

        StandardLayer standardLayer = new StandardLayer(
                30, convolutionLayer, Sigmoid.getInstance(), new GaussianInitializer(), 0);
        StandardLayer outputLayer = new StandardLayer(
                10, standardLayer, Sigmoid.getInstance(), new GaussianInitializer(), 0);

        NeuralNetwork nn = new NeuralNetwork(inputLayer, CrossEntropy.getInstance(), convolutionLayer, standardLayer, outputLayer);
        System.out.println(".");

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

            ImageWriter.write(image, 28, 28, e.input);
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

    private static double checkLabels(NeuralNetwork nn, Data.Example[] data) {
        double correct = 0;
        for(int i = 0; i < data.length; i++) {
            Data.Example e = data[i];
            nn.run(e.input);
            int predicted = getLabel(nn.predicted);
            if (predicted == getLabel(e.output)) correct += 1;
        }

        return correct / data.length;
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
