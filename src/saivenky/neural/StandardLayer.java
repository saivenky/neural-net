package saivenky.neural;

import saivenky.neural.activation.ActivationFunction;
import saivenky.neural.neuron.NeuronInitializer;

/**
 * Created by saivenky on 1/31/17.
 */
public class StandardLayer extends Layer implements IDropoutLayer {
    private double dropoutRate;
    private int[] nonDropout;

    public StandardLayer(
            int neuronCount, ILayer previousLayer, ActivationFunction activationFunction, NeuronInitializer neuronInitializer, double dropoutRate) {
        super(new ThreadedNeuronSet(new Neuron[neuronCount]));
        initializeNeurons(neuronInitializer, previousLayer.getNeurons(), activationFunction);
        this.dropoutRate = dropoutRate;
        setDropoutRate(this.dropoutRate);
    }

    private void initializeNeurons(
            NeuronInitializer neuronInitializer, NeuronSet previousLayerNeurons, ActivationFunction activationFunction) {
        for(int i = 0; i < neurons.size(); i++) {
            neurons.set(i, new Neuron(neuronInitializer, previousLayerNeurons, activationFunction));
        }
    }

    private static double signalScaleFromDropout(double dropoutRate) {
        return (1 - dropoutRate);
    }

    @Override
    public void run() {
        runWithoutDropout();
    }

    public void runWithoutDropout() {
        double scale = signalScaleFromDropout(dropoutRate);
        neurons.activateScaled(scale);
    }

    public void setDropoutRate(double dropoutRate) {
        this.dropoutRate = dropoutRate;
        int nonDropoutLength = neurons.size() - (int)(neurons.size() * dropoutRate);
        nonDropout = new int[nonDropoutLength];
        reselectDropout();
        neurons.select(nonDropout);
    }

    public void reselectDropout() {
        Vector.select(nonDropout, neurons.size());
    }
}
