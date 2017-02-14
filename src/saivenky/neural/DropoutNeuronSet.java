package saivenky.neural;

/**
 * Created by saivenky on 1/30/17.
 */
public class DropoutNeuronSet extends NeuronSet {
    protected int[] selected;

    public DropoutNeuronSet(INeuron[] neurons) {
        super(neurons);
        selected = new int[neurons.length];
        Vector.select(selected, neurons.length);
    }

    void select(int[] selected) {
        this.selected = selected;
    }

    public void activate() {
        for(int i : selected) {
            neurons[i].activate();
        }
    }

    public void backpropagate(boolean backpropagateToInputNeurons) {
        for(int i : selected) {
            neurons[i].backpropagate(backpropagateToInputNeurons);
        }
    }

    public void gradientDescent(double rate) {
        for(int i : selected) {
            neurons[i].gradientDescent(rate);
        }
    }

    public void addSignalCostGradient(double[] weight, double cost) {
        for(int i : selected) {
            neurons[i].addToSignalCostGradient(weight[i], cost);
        }
    }

    public double affine(double[] weight, double bias) {
        double sum = 0;
        for(int i : selected) {
            sum += weight[i] * neurons[i].getActivation();
        }

        return sum + bias;
    }

    public void addToWeightError(double[] weightError, double cost) {
        for(int i : selected) {
            weightError[i] += neurons[i].getActivation() * cost;
        }
    }
}
