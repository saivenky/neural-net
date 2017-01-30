package saivenky.neural.mnist;

import saivenky.neural.Data;
import saivenky.neural.NeuralNetwork;
import saivenky.neural.NeuralNetworkTrainer;
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

        int[] layers = {28 * 28, 120, 10};
        double[] dropouts = {0.25};

        NeuralNetwork nn = new NeuralNetwork(layers, Sigmoid.getInstance(), CrossEntropy.getInstance(), new GaussianInitializer());
        nn.setDropouts(dropouts);

        NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(nn, trainingData);

        int batchSize = 60;
        double learningRate = 3.0;
        int epochs = 15;

        for(int i = 0; i < epochs; i++) {
            trainer.train(learningRate, batchSize);
            //System.out.println("train data correct: " + checkLabels(nn, trainingData));
            System.out.println("test data correct: " + checkLabels(nn, testData, testLabels));
        }

        File outputDir = new File("/home/saivenky/mnist-testing-result");
        System.out.println("test data correct: " + checkLabelsAndWriteIncorrect(nn, testData, testLabels, outputDir));
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

    private static double checkLabels(NeuralNetwork nn, Data.Example[] data, int[] labels) {
        double correct = 0;
        if(data.length != labels.length) throw  new RuntimeException("Data and label length mismatch");
        for(int i = 0; i < data.length; i++) {
            Data.Example e = data[i];
            nn.run(e.input);
            int predicted = getLabel(nn.predicted);
            if (predicted == labels[i]) correct += 1;
        }

        return correct / data.length;
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
