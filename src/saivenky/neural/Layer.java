package saivenky.neural;

/**
 * Created by saivenky on 1/26/17.
 */
public class Layer {
    public NeuronSet neurons;
    double dropoutRate;
    int[] nonDropout;

    Layer(NeuronSet neurons) {
        this.neurons = neurons;
    }

    final void setDropoutRate(double dropoutRate) {
        this.dropoutRate = dropoutRate;
        int nonDropoutLength = neurons.size() - (int)(neurons.size() * dropoutRate);
        nonDropout = new int[nonDropoutLength];
        Vector.select(nonDropout, neurons.size());
        neurons.select(nonDropout);
    }

    final void reselectDropout() {
        Vector.select(nonDropout, neurons.size());
    }

    void run() {
        neurons.activate();
    }

    private static double signalScaleFromDropout(double dropoutRate) {
        return (1 - dropoutRate);
    }

    void runScaled(double inputDropoutRate) {
        double scale = signalScaleFromDropout(inputDropoutRate);
        neurons.activateScaled(scale);
    }

    void backpropagate() {
        neurons.forSelected(propagateToInputNeurons);
    }

    void updateGradient() {
        neurons.forSelected(propagateToProperties);
    }

    void gradientDescent(double rate) {
        neurons.forSelected(new NeuronSet.NeuronAction() {
            @Override
            void f(Neuron neuron, int i) {
                neuron.update(rate);
            }
        });
    }

    private static NeuronSet.NeuronAction propagateToInputNeurons = new NeuronSet.NeuronAction() {
        @Override
        void f(Neuron neuron, int i) {
            neuron.propagateToInputNeurons();
        }
    };

    private static NeuronSet.NeuronAction propagateToProperties = new NeuronSet.NeuronAction() {
        @Override
        void f(Neuron neuron, int i) {
            neuron.propagateToProperties();
        }
    };
}
