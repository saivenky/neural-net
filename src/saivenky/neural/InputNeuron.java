package saivenky.neural;

/**
 * Created by saivenky on 1/31/17.
 */
public class InputNeuron implements INeuron {
    private double activation;

    public InputNeuron() {
        activation = 0;
    }

    void setActivation(double activation) {
        this.activation = activation;
    }

    @Override
    public double getActivation() {
        return activation;
    }

    @Override
    public void activate() {

    }

    @Override
    public void activateScaled(double scale) {

    }

    @Override
    public void gradientDescent(double rate) {

    }

    @Override
    public void setSignalCostGradient(double cost) {

    }

    @Override
    public void addToSignalCostGradient(double cost) {

    }

    @Override
    public void addToSignalCostGradient(double weight, double cost) {

    }

    @Override
    public void backpropagate(boolean backpropagateToInputNeurons) {

    }
}
