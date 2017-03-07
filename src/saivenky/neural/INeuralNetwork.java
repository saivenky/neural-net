package saivenky.neural;

/**
 * Created by saivenky on 3/4/17.
 */
public interface INeuralNetwork {
    double[][] getPredicted();
    void run(double[][] input);
    void update(double rate);
    void train(double[][] input, double[][] output);
}
