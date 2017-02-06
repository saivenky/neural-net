package saivenky.neural;

/**
 * Created by saivenky on 1/30/17.
 */
public class NeuronSet {
    INeuron[] neurons;
    public int[] selected;

    NeuronSet(INeuron[] neurons) {
        this.neurons = neurons;
        selected = new int[neurons.length];
        Vector.select(selected, neurons.length);
    }

    void select(int[] selected) {
        this.selected = selected;
    }

    public INeuron get(int index) {
        return neurons[index];
    }

    void set(int index, INeuron neuron) {
        neurons[index] = neuron;
    }

    public int size() {
        return neurons.length;
    }

    public void activate() {
        for(int i : selected) {
            neurons[i].activate();
        }
    }

    public void activateScaled(double scale) {
        for(int i = 0; i < neurons.length; i++) {
            neurons[i].activateScaled(scale);
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

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append('[');
        for(int i = 0; i < neurons.length - 1; i++) {
            builder.append(neurons[i].getActivation() + ", ");
        }
        builder.append(neurons[neurons.length - 1].getActivation() + "]");
        return builder.toString();
    }
}
