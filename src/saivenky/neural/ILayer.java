package saivenky.neural;

/**
 * Created by saivenky on 2/5/17.
 */
public interface ILayer {
    NeuronSet getNeurons();

    void run();

    void feedforward();

    void backpropagate();

    void updateGradient();

    void gradientDescent(double rate);

    void setSignalCostGradient(double[] cost);

    void multiplySignalCostGradientByActivation1();
}

