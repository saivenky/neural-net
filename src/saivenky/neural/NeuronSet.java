package saivenky.neural;

/**
 * Created by saivenky on 1/30/17.
 */
public class NeuronSet {
    Neuron[] neurons;
    public int[] selected;

    static abstract class NeuronAction {
        abstract void f(Neuron neuron, int i);
    }

    NeuronSet(Neuron[] neurons) {
        this.neurons = neurons;
        selected = null;
    }

    void select(int[] selected) {
        this.selected = selected;
    }

    public Neuron get(int index) {
        return neurons[index];
    }

    void set(int index, Neuron neuron) {
        neurons[index] = neuron;
    }

    public int size() {
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

    void forAll(NeuronAction action) {
        for (int i = 0; i < neurons.length; i++) {
            action.f(neurons[i], i);
        }
    }

    public void activate() {
        forSelected(signalForSelected);
    }

    public void activateScaled(double scale) {
        forAll(signalScaled.withScale(scale));
    }

    private static NeuronAction signalForSelected = new NeuronAction() {
        @Override
        void f(Neuron neuron, int i) {
            neuron.signalForSelected();
        }
    };

    private static abstract class NeuronActionWithScale extends NeuronAction {
        protected double scale;

        NeuronAction withScale(double scale) {
            this.scale = scale;
            return this;
        }
    }

    private static NeuronActionWithScale signalScaled = new NeuronActionWithScale() {
        @Override
        void f(Neuron neuron, int i) {
            neuron.signalScaled(scale);
        }
    };
}
