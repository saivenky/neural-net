package saivenky.neural;

/**
 * Created by saivenky on 2/7/17.
 */
public class FilterDimensionCalculator {
    public static int calculateOutputSize(int inputSize, int filterSize, int stride, int padding) {
        int temp = inputSize - filterSize + 2 * padding;
        if (temp % stride == 0) return temp / stride + 1;
        throw new RuntimeException("Filter and stride size do not create round output number.");
    }
}
