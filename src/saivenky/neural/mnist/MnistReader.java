package saivenky.neural.mnist;

import saivenky.neural.Data;
import saivenky.neural.image.DataUtils;

import java.io.*;
import java.util.zip.GZIPInputStream;

/**
 * Created by saivenky on 1/28/17.
 */
class MnistReader {
    private static float[] readImage(InputStream stream, int rows, int columns) {
        int totalBytes = rows * columns;
        float[] imagePixels = new float[totalBytes];
        byte[] rawBytes = new byte[totalBytes];
        int numBytesRead = 0;

        try {
            int tries = 0;
            while (numBytesRead != totalBytes) {
                numBytesRead += stream.read(rawBytes, numBytesRead, rawBytes.length - numBytesRead);
                tries++;
                if (tries > 3) {
                    throw new RuntimeException("unable to read all pixels for image within 3 tries");
                }
            }
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

    static File getRealFile(String filePath) throws FileNotFoundException {
        File file = new File(filePath);
        if(!file.exists()) throw new FileNotFoundException(filePath);
        return file;
    }

    private static float[][] readImages(File imagesFile) throws IOException {
        DataInputStream dis = new DataInputStream(new GZIPInputStream(new FileInputStream(imagesFile)));
        dis.readInt(); //magic num
        int numImages = dis.readInt();
        int rows = dis.readInt();
        int columns = dis.readInt();

        float[][] images = new float[numImages][];
        for(int i = 0; i < numImages; i++) {
            images[i] = readImage(dis, rows, columns);
        }

        return images;
    }

    private static int[] readLabels(File labelsFile) throws IOException {
        DataInputStream dis = new DataInputStream(new GZIPInputStream(new FileInputStream(labelsFile)));
        dis.readInt(); //magic num
        int numLabels = dis.readInt();

        int[] labels = new int[numLabels];
        for(int i = 0; i < numLabels; i++) {
            labels[i] = dis.readUnsignedByte();
        }

        return labels;
    }

    private static float[] toOutput(int label) {
        float[] output = new float[10];
        output[label] = 1;
        return output;
    }

    static Data.Example[] loadMnist(String imagesFilePath, String labelsFilePath) throws IOException {
        File imagesFile = getRealFile(imagesFilePath);
        File labelsFile = getRealFile(labelsFilePath);
        System.out.printf("loading MNIST (images: %s, labels: %s)", imagesFile.getName(), labelsFile.getName());
        float[][] images = readImages(imagesFile);
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
}
