package saivenky.neural;

/**
 * Created by saivenky on 1/30/17.
 */
public class NeuronSet {
    Neuron[] neurons;
    int[] selected;
    double[] activation;
    double[] activation1;

    public static abstract class NeuronAction {
        abstract void f(Neuron neuron, int i);
    }

    public NeuronSet(Neuron[] neurons) {
        this.neurons = neurons;
        selected = null;
        activation = new double[neurons.length];
        activation1 = new double[neurons.length];
    }

    public void select(int[] selected) {
        this.selected = selected;
    }

    public void clearSelection() {
        select(null);
    }

    Neuron get(int index) {
        return neurons[index];
    }

    void set(int index, Neuron neuron) {
        neurons[index] = neuron;
    }

    int size() {
        return neurons.length;
    }

    public void forSelected(NeuronAction action) {
        if(selected == null) {
            forAll(action);
            return;
        }

        for(int i : selected) {
            action.f(neurons[i], i);
        }
    }

    public void forAll(NeuronAction action) {
        for (int i = 0; i < neurons.length; i++) {
            action.f(neurons[i], i);
        }
    }
}
