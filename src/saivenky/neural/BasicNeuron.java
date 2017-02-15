package saivenky.neural;

import saivenky.neural.activation.ActivationFunction;

/**
 * Created by saivenky on 2/14/17.
 */
public class BasicNeuron implements INeuron {
    private ActivationFunction activationFunction;
    private double signal;
    private double activation;
    private double activation1;
    private double error;

    public BasicNeuron(ActivationFunction activationFunction) {

        this.activationFunction = activationFunction;
    }

    public void setSignal(double signal) {
        this.signal = signal;
    }

    public double getError() {
        return error;
    }

    @Override
    public double getActivation() {
        return activation;
    }

    @Override
    public void activate() {
        activation = activationFunction.f(signal);
        activation1 = activationFunction.f1(signal);
    }

    @Override
    public void activateScaled(double scale) {
        activate();
    }

    @Override
    public void gradientDescent(double rate) {

    }

    @Override
    public void setSignalCostGradient(double cost) {
        error = cost;
    }

    @Override
    public void addToSignalCostGradient(double cost) {
        error += cost;
    }

    @Override
    public void addToSignalCostGradient(double weight, double cost) {
        error += weight * cost;
    }

    @Override
    public void backpropagate(boolean backpropagateToInputNeurons) {
        error *= activation1;
    }
}
