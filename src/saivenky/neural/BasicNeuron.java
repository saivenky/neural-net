package saivenky.neural;

/**
 * Created by saivenky on 2/14/17.
 */
public class BasicNeuron implements INeuron {
    private double signal;
    private double error;

    public void setSignal(double signal) {
        this.signal = signal;
    }

    public double getError() {
        return error;
    }

    @Override
    public double getActivation() {
        return signal;
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
    public synchronized void setSignalCostGradient(double cost) {
        error = cost;
    }

    @Override
    public synchronized void addToSignalCostGradient(double cost) {
        error += cost;
    }

    @Override
    public synchronized void addToSignalCostGradient(double weight, double cost) {
        error += weight * cost;
    }

    @Override
    public void backpropagate(boolean backpropagateToInputNeurons) {
    }
}
