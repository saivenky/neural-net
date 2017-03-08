package saivenky.neural;

/**
 * Created by saivenky on 3/4/17.
 */
public interface INeuralNetwork {
    float[][] getPredicted();
    void run(float[][] input);
    void update(float rate);
    void train(float[][] input, float[][] output);
}
