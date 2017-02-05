package saivenky.neural;

/**
 * Created by saivenky on 2/5/17.
 */
public interface INeuron {
    double getActivation();

    void activate();

    void activateScaled(double scale);

    void update(double rate);

    void addToSignalCostGradient(double weight, double cost);

    void multiplyByActivation1();

    void propagateToInputNeurons();

    void propagateToProperties();
}
